from __future__ import annotations

import logging
from uuid import UUID

from .config import Settings
from .entity_vector_store import EntityVectorStore
from .graph_repository import Neo4jRepository
from .ingestion_models import DeleteDocumentResponse
from .ingestion_repository import IngestionRepository
from .query_cache import QueryCache
from .secrets import resolve_neo4j_password

logger = logging.getLogger(__name__)


def delete_document(doc_id: UUID, settings: Settings) -> DeleteDocumentResponse:
    """Cascade-delete a document and all derived data.

    Deletion order (each step is best-effort so later steps still run):
      1. Resolve chunk_ids for entity dereferencing
      2. Neo4j entities & relations (by source_chunk_ids)
      3. pgvector entities & relations (by source_chunk_ids, FK cascade)
      4. PostgreSQL chunks (all versions)
      5. OpenSearch documents (by doc_id term)
      6. L2 query cache invalidation
      7. PostgreSQL document record
    """

    repo = IngestionRepository(settings)
    doc_id_str = str(doc_id)

    # Resolve chunk_ids before deleting chunks
    chunk_ids = repo.get_chunk_ids_for_doc(doc_id)

    entities_deleted = 0
    relations_deleted = 0

    # --- Neo4j cleanup ---
    if settings.enable_neo4j and chunk_ids:
        neo4j_repo: Neo4jRepository | None = None
        try:
            neo4j_password = resolve_neo4j_password(settings)
            neo4j_repo = Neo4jRepository(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=neo4j_password,
                database=settings.neo4j_database,
            )
            relations_deleted += neo4j_repo.delete_relations_by_source_chunks(chunk_ids)
            entities_deleted += neo4j_repo.delete_entities_by_source_chunks(chunk_ids)
            neo4j_repo.cleanup_orphan_relations()
        except Exception:
            logger.error("Neo4j cleanup failed for doc %s; continuing", doc_id, exc_info=True)
        finally:
            if neo4j_repo is not None:
                try:
                    neo4j_repo.close()
                except Exception:
                    logger.warning("Failed to close Neo4j connection", exc_info=True)

    # --- pgvector entity cleanup ---
    if chunk_ids:
        try:
            vector_store = EntityVectorStore(settings)
            pg_entity_count = vector_store.delete_entities_by_source_chunks(chunk_ids)
            entities_deleted += pg_entity_count
        except Exception:
            logger.error(
                "pgvector entity cleanup failed for doc %s; continuing", doc_id, exc_info=True
            )

    # --- PostgreSQL chunks ---
    chunks_deleted = repo.delete_all_chunks_for_doc(doc_id)

    # --- OpenSearch ---
    opensearch_deleted = 0
    try:
        opensearch_deleted = repo.delete_opensearch_docs_by_doc_id(doc_id)
    except Exception:
        logger.error("OpenSearch cleanup failed for doc %s; continuing", doc_id, exc_info=True)

    # --- L2 query cache ---
    cache_invalidated = False
    try:
        cache = QueryCache(settings)
        cache.invalidate_by_doc(doc_id_str)
        cache_invalidated = True
    except Exception:
        logger.error("Cache invalidation failed for doc %s; continuing", doc_id, exc_info=True)

    # --- Document record ---
    repo.delete_document_record(doc_id)

    return DeleteDocumentResponse(
        doc_id=doc_id_str,
        chunks_deleted=chunks_deleted,
        entities_deleted=entities_deleted,
        relations_deleted=relations_deleted,
        opensearch_deleted=opensearch_deleted,
        cache_invalidated=cache_invalidated,
    )
