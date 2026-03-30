"""Document ingestion pipeline orchestrator.

Coordinates S3 download → parse → chunk → embed → store (PG + OpenSearch) → status update.
When entity extraction is enabled (Phase 2+), also runs:
  extract entities/relations → dedup/merge → embed → store (Neo4j + pgvector).
All operations are synchronous (boto3, SQLAlchemy).
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING
from uuid import UUID

import boto3

from .chunker import Chunk, chunk_document
from .config import Settings
from .document_parser import parse_document
from .entity_extraction import EntityDeduplicator, EntityExtractor
from .entity_extraction_models import ChunkExtractionResult, ExtractedEntity, ExtractedRelation
from .entity_vector_store import EntityVectorStore
from .graph_repository import Neo4jRepository
from .ingestion_models import ChunkRecord, DocumentRecord, IngestionResult, UploadMetadata
from .ingestion_repository import IngestionRepository
from .qwen_client import QwenClient
from .secrets import resolve_neo4j_password

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _run_entity_extraction_pipeline(
    chunk_records: list[ChunkRecord],
    doc_id: UUID,
    qwen_client: QwenClient,
    settings: Settings,
) -> tuple[int, int]:
    """Run entity extraction, dedup/merge, embedding, and storage.

    Pipeline:
        1. Extract entities/relations from each chunk via LLM
        2. Deduplicate and merge across chunks
        3. Generate embeddings for entity descriptions and relation evidence
        4. Store to Neo4j (graph traversal) and pgvector (similarity search)

    Each storage backend is independent — a Neo4j failure does not block pgvector
    writes, and vice versa. Individual chunk extraction failures are logged and
    skipped so the remaining chunks can still be processed.

    Args:
        chunk_records: Chunk records already persisted in PG (with embeddings).
        doc_id: The document UUID that owns these chunks.
        qwen_client: QwenClient instance for LLM extraction and embedding calls.
        settings: Runtime settings (Neo4j URI, batch sizes, feature flags, etc.).

    Returns:
        (entity_count, relation_count) — totals successfully stored.
    """
    # --- 1. Extract entities/relations from each chunk ---
    extractor = EntityExtractor(qwen_client, gleaning_rounds=settings.extraction_gleaning_rounds)
    chunk_results: list[ChunkExtractionResult] = []

    for chunk_rec in chunk_records:
        chunk_id = f"{doc_id}_chunk_{chunk_rec.chunk_index}"
        try:
            result, _trace = extractor.extract(chunk_id, str(doc_id), chunk_rec.chunk_text)
            chunk_results.append(result)
        except Exception:
            logger.warning(
                "Entity extraction failed for chunk %s, skipping",
                chunk_id,
                exc_info=True,
            )

    if not chunk_results:
        logger.info("No successful chunk extractions for doc %s; skipping entity pipeline", doc_id)
        return 0, 0

    # --- 2. Deduplicate and merge entities/relations across chunks ---
    deduplicator = EntityDeduplicator(qwen_client)
    merged_entities: list[ExtractedEntity] = deduplicator.merge_entities(chunk_results)
    merged_relations: list[ExtractedRelation] = deduplicator.merge_relations(
        chunk_results, merged_entities
    )

    if not merged_entities and not merged_relations:
        logger.info("No entities or relations after dedup for doc %s", doc_id)
        return 0, 0

    # --- 3. Generate embeddings for entities and relations ---
    batch_size = settings.entity_extraction_embed_batch_size

    entity_embeddings: list[list[float]] = []
    if merged_entities:
        entity_texts = [f"{e.name} {e.type.value} {e.description}" for e in merged_entities]
        for batch_start in range(0, len(entity_texts), batch_size):
            batch = entity_texts[batch_start : batch_start + batch_size]
            entity_embeddings.extend(qwen_client.embedding(batch))  # type: ignore[arg-type]

    relation_embeddings: list[list[float]] = []
    if merged_relations:
        relation_texts = [f"{r.type.value} {r.evidence}" for r in merged_relations]
        for batch_start in range(0, len(relation_texts), batch_size):
            batch = relation_texts[batch_start : batch_start + batch_size]
            relation_embeddings.extend(qwen_client.embedding(batch))  # type: ignore[arg-type]

    # --- 4a. Store to Neo4j (graph traversal) ---
    entity_count = 0
    relation_count = 0

    if settings.enable_neo4j:
        neo4j_repo: Neo4jRepository | None = None
        try:
            neo4j_password = resolve_neo4j_password(settings)
            neo4j_repo = Neo4jRepository(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=neo4j_password,
                database=settings.neo4j_database,
            )
            stats = neo4j_repo.upsert_batch(merged_entities, merged_relations)
            entity_count = stats.get("entities_written", 0)
            relation_count = stats.get("relations_written", 0)
            logger.info(
                "Neo4j: wrote %d entities, %d relations for doc %s",
                entity_count,
                relation_count,
                doc_id,
            )
        except Exception:
            logger.error(
                "Neo4j storage failed for doc %s; continuing with pgvector",
                doc_id,
                exc_info=True,
            )
        finally:
            if neo4j_repo is not None:
                try:
                    neo4j_repo.close()
                except Exception:
                    logger.warning("Failed to close Neo4j connection", exc_info=True)

    # --- 4b. Store to pgvector (similarity search) ---
    try:
        vector_store = EntityVectorStore(settings)

        old_to_new_id: dict[str, str] = {}
        entity_dicts: list[dict] = []
        for idx, entity in enumerate(merged_entities):
            dedup_key = entity.canonical_key or entity.name.lower()
            stable_id = hashlib.sha256(f"{dedup_key}::{entity.type.value}".encode()).hexdigest()[
                :16
            ]
            old_to_new_id[entity.entity_id] = stable_id
            entity_dicts.append(
                {
                    "entity_id": stable_id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "description": entity.description,
                    "embedding": entity_embeddings[idx],
                    "canonical_key": entity.canonical_key,
                    "aliases": entity.aliases,
                    "confidence": entity.confidence,
                    "source_chunk_ids": entity.source_chunk_ids,
                }
            )

        relation_dicts: list[dict] = []
        for idx, relation in enumerate(merged_relations):
            src_id = old_to_new_id.get(relation.source_entity_id, relation.source_entity_id)
            tgt_id = old_to_new_id.get(relation.target_entity_id, relation.target_entity_id)
            relation_id = f"{src_id}__{relation.type.value}__{tgt_id}"
            relation_dicts.append(
                {
                    "relation_id": relation_id,
                    "source_entity_id": src_id,
                    "target_entity_id": tgt_id,
                    "type": relation.type.value,
                    "evidence": relation.evidence,
                    "embedding": relation_embeddings[idx],
                    "confidence": relation.confidence,
                    "weight": relation.weight,
                    "source_chunk_ids": relation.source_chunk_ids,
                }
            )

        pgv_entity_count = vector_store.batch_upsert_entities(entity_dicts) if entity_dicts else 0
        pgv_relation_count = (
            vector_store.batch_upsert_relations(relation_dicts) if relation_dicts else 0
        )
        logger.info(
            "pgvector: wrote %d entities, %d relations for doc %s",
            pgv_entity_count,
            pgv_relation_count,
            doc_id,
        )

        # Use pgvector counts as fallback if Neo4j was disabled or failed
        if not settings.enable_neo4j or entity_count == 0:
            entity_count = pgv_entity_count
        if not settings.enable_neo4j or relation_count == 0:
            relation_count = pgv_relation_count

    except Exception:
        logger.error(
            "pgvector storage failed for doc %s; entity counts may be incomplete",
            doc_id,
            exc_info=True,
        )

    return entity_count, relation_count


def ingest_document(
    s3_bucket: str,
    s3_key: str,
    upload_metadata: UploadMetadata,
    settings: Settings,
) -> IngestionResult:
    """Orchestrate the full document ingestion pipeline.

    Steps:
        1. Create ingestion run record
        2. Download file bytes from S3
        3. Parse the document
        4. Chunk into overlapping text segments
        5. Generate embeddings in batches
        6. Upsert document record
        7. Delete stale chunks for this doc/version
        8. Insert new chunk records
        9. Index chunks to OpenSearch
        10. Mark run as succeeded and return result

    Args:
        s3_bucket: S3 bucket name where the file is stored.
        s3_key: S3 object key (may include path prefix).
        upload_metadata: Metadata attached to the upload request.
        settings: Runtime settings from environment.

    Returns:
        IngestionResult with run_id, doc_id, status, chunks_created, and optional error.
    """
    repo = IngestionRepository(settings)
    qwen_client = QwenClient(settings)

    run_id: UUID = repo.create_ingestion_run(
        source_type="file",
        config={
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "doc_version": upload_metadata.doc_version,
        },
    )

    doc_id: UUID = UUID(int=0)  # sentinel; replaced after upsert_document succeeds

    try:
        s3 = boto3.client("s3")
        file_bytes: bytes = s3.get_object(Bucket=s3_bucket, Key=s3_key)["Body"].read()

        filename = s3_key.split("/")[-1]
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        parsed = parse_document(file_bytes, filename)

        chunks: list[Chunk] = chunk_document(
            parsed,
            chunk_size=settings.ingestion_chunk_size,
            chunk_overlap=settings.ingestion_chunk_overlap,
            min_chunk_size=settings.ingestion_chunk_min_size,
        )

        embeddings: list[list[float]] = []
        if chunks:
            batch_size = settings.ingestion_embed_batch_size
            for batch_start in range(0, len(chunks), batch_size):
                batch_texts = [c.chunk_text for c in chunks[batch_start : batch_start + batch_size]]
                # embedding() returns list[list[float]] when input is list[str]
                batch_embeddings = qwen_client.embedding(batch_texts)
                embeddings.extend(batch_embeddings)  # type: ignore[arg-type]

        source_uri = upload_metadata.source_uri or f"s3://{s3_bucket}/{s3_key}"
        title = upload_metadata.title or parsed.title

        doc_record = DocumentRecord(
            source_type="file",
            source_uri=source_uri,
            title=title,
            lang=upload_metadata.lang,
            category=upload_metadata.category,
            mime_type=parsed.mime_type,
            content_hash=content_hash,
            doc_version=upload_metadata.doc_version,
            published_year=upload_metadata.published_year,
            published_month=upload_metadata.published_month,
            author=upload_metadata.author,
            tags=upload_metadata.tags,
            metadata={},
            run_id=run_id,
        )
        doc_id = repo.upsert_document(doc_record)

        repo.delete_chunks_for_doc(doc_id, upload_metadata.doc_version)

        chunk_records: list[ChunkRecord] = [
            ChunkRecord(
                doc_id=doc_id,
                doc_version=upload_metadata.doc_version,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                token_count=chunk.token_count,
                citation_url=source_uri,
                citation_title=title,
                citation_year=upload_metadata.published_year,
                citation_month=upload_metadata.published_month,
                embedding=embeddings[i],
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_id=chunk.section_id,
                anchor_id=chunk.anchor_id,
                metadata={},
                run_id=run_id,
            )
            for i, chunk in enumerate(chunks)
        ]

        if chunk_records:
            repo.batch_insert_chunks(chunk_records)
            repo.bulk_index_opensearch(chunk_records)

        # --- Entity extraction pipeline (Phase 2+) ---
        entity_count = 0
        relation_count = 0
        if settings.enable_entity_extraction and chunk_records:
            try:
                entity_count, relation_count = _run_entity_extraction_pipeline(
                    chunk_records=chunk_records,
                    doc_id=doc_id,
                    qwen_client=qwen_client,
                    settings=settings,
                )
            except Exception:
                logger.error(
                    "Entity extraction pipeline failed for doc %s; "
                    "chunk ingestion already succeeded — continuing",
                    doc_id,
                    exc_info=True,
                )

        repo.complete_ingestion_run(run_id, "succeeded")

        return IngestionResult(
            run_id=run_id,
            doc_id=doc_id,
            status="succeeded",
            chunks_created=len(chunk_records),
            entity_count=entity_count,
            relation_count=relation_count,
        )

    except Exception as exc:
        logger.exception("Ingestion failed for s3://%s/%s: %s", s3_bucket, s3_key, exc)
        try:
            repo.complete_ingestion_run(run_id, "failed", notes=str(exc))
        except Exception:
            logger.exception("Failed to mark ingestion run %s as failed", run_id)
        return IngestionResult(
            run_id=run_id,
            doc_id=doc_id,
            status="failed",
            chunks_created=0,
            error=str(exc),
        )
