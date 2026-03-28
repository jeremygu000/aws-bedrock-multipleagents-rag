"""Document ingestion pipeline orchestrator.

Coordinates S3 download → parse → chunk → embed → store (PG + OpenSearch) → status update.
All operations are synchronous (boto3, SQLAlchemy).
"""

import hashlib
import logging
from uuid import UUID

import boto3

from .chunker import Chunk, chunk_document
from .config import Settings
from .document_parser import parse_document
from .ingestion_models import ChunkRecord, DocumentRecord, IngestionResult, UploadMetadata
from .ingestion_repository import IngestionRepository
from .qwen_client import QwenClient

logger = logging.getLogger(__name__)


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

        repo.complete_ingestion_run(run_id, "succeeded")

        return IngestionResult(
            run_id=run_id,
            doc_id=doc_id,
            status="succeeded",
            chunks_created=len(chunk_records),
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
