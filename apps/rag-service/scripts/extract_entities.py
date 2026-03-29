"""Standalone entity extraction script for already-ingested documents.

Reads documents and their chunks from kb_documents/kb_chunks, runs the entity
extraction pipeline (LLM extract → dedup/merge → embed → store to Neo4j + pgvector),
and reports progress.

Designed to backfill entities for documents that were ingested with --skip-entities.

Usage:
    cd apps/rag-service && python -m scripts.extract_entities [OPTIONS]

Options:
    --limit N       Only process first N documents (for testing)
    --doc-id UUID   Process a single document by ID (repeatable)
    --category CAT  Filter documents by category (e.g. pages, news)
    --concurrency N Parallel document processing (default: 1, sequential)
    --force         Re-extract even for docs that already have entities
    --dry-run       Show what would be processed but don't run extraction
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text

from app.config import Settings, get_settings
from app.ingestion import _run_entity_extraction_pipeline
from app.ingestion_models import ChunkRecord
from app.ingestion_repository import IngestionRepository
from app.qwen_client import QwenClient

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.extract_entities")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DocInfo:
    """Lightweight document record for entity extraction."""

    doc_id: UUID
    title: str
    category: str
    source_uri: str


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------


def get_documents(
    repo: IngestionRepository,
    *,
    category: str | None = None,
    doc_ids: list[str] | None = None,
    limit: int | None = None,
) -> Iterator[DocInfo]:
    """Fetch documents from kb_documents as a streaming iterator."""

    conditions: list[str] = []
    params: dict = {}

    if doc_ids:
        conditions.append("doc_id = ANY(:doc_ids)")
        params["doc_ids"] = [UUID(d) for d in doc_ids]

    if category:
        conditions.append("category = :category")
        params["category"] = category

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    limit_clause = ""
    if limit is not None and limit > 0:
        limit_clause = f"LIMIT {limit}"

    sql = f"""
        SELECT doc_id, title, category, source_uri
        FROM kb_documents
        {where_clause}
        ORDER BY created_at ASC
        {limit_clause}
    """

    engine = repo._get_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), params)
        for row in result:
            yield DocInfo(
                doc_id=row.doc_id,
                title=row.title,
                category=row.category,
                source_uri=row.source_uri,
            )


def get_chunks_for_doc(repo: IngestionRepository, doc_id: UUID) -> Iterator[ChunkRecord]:
    """Stream chunks for a document from kb_chunks, yielding one at a time."""

    sql = """
        SELECT doc_id, doc_version, chunk_index, chunk_text, token_count,
               citation_url, citation_title, citation_year, citation_month,
               page_start, page_end, section_id, anchor_id, embedding, metadata
        FROM kb_chunks
        WHERE doc_id = :doc_id
        ORDER BY chunk_index ASC
    """

    engine = repo._get_engine()
    with engine.begin() as conn:
        result = conn.execute(text(sql), {"doc_id": doc_id})
        for row in result:
            # Parse embedding from pgvector string "[0.1,0.2,...]" → list[float]
            embedding_raw = row.embedding
            if isinstance(embedding_raw, str):
                embedding = [float(x) for x in embedding_raw.strip("[]").split(",")]
            elif isinstance(embedding_raw, (list, tuple)):
                embedding = list(embedding_raw)
            else:
                embedding = []

            yield ChunkRecord(
                doc_id=row.doc_id,
                doc_version=row.doc_version,
                chunk_index=row.chunk_index,
                chunk_text=row.chunk_text,
                token_count=row.token_count,
                citation_url=row.citation_url,
                citation_title=row.citation_title,
                citation_year=row.citation_year,
                citation_month=row.citation_month,
                embedding=embedding,
                page_start=row.page_start,
                page_end=row.page_end,
                section_id=row.section_id,
                anchor_id=row.anchor_id,
                metadata=row.metadata or {},
            )


def get_docs_with_entities(repo: IngestionRepository) -> set[UUID]:
    # source_chunk_ids contain '{doc_id}_chunk_{index}' — extract doc_id prefix
    sql = """
        SELECT DISTINCT
            CAST(split_part(chunk_id, '_chunk_', 1) AS UUID) AS doc_id
        FROM (
            SELECT unnest(source_chunk_ids) AS chunk_id
            FROM kb_entities
        ) sub
    """

    engine = repo._get_engine()
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(sql)).fetchall()
        return {row.doc_id for row in rows}
    except Exception:
        logger.warning(
            "Could not query existing entities (kb_entities may be empty or not exist)",
            exc_info=True,
        )
        return set()


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def process_single_doc(
    doc: DocInfo,
    repo: IngestionRepository,
    qwen_client: QwenClient,
    settings: Settings,
    index: int,
    total: int,
) -> tuple[int, int, float]:
    t0 = time.monotonic()
    logger.info(
        "[%d/%d] Processing doc %s: %s",
        index,
        total,
        doc.doc_id,
        doc.title[:80],
    )

    chunks = list(get_chunks_for_doc(repo, doc.doc_id))
    if not chunks:
        logger.warning("  No chunks found for doc %s — skipping", doc.doc_id)
        return 0, 0, time.monotonic() - t0

    logger.info("  Found %d chunks", len(chunks))

    entity_count, relation_count = _run_entity_extraction_pipeline(
        chunk_records=chunks,
        doc_id=doc.doc_id,
        qwen_client=qwen_client,
        settings=settings,
    )

    elapsed = time.monotonic() - t0
    logger.info(
        "  Done: %d entities, %d relations (%.1fs)",
        entity_count,
        relation_count,
        elapsed,
    )
    return entity_count, relation_count, elapsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract entities from already-ingested documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N documents (for testing)",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        default=None,
        help="Process specific document(s) by UUID (repeatable)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter documents by category (e.g. pages, news)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel document processing threads (default: 1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-extract even for docs that already have entities",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be processed without running extraction",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings = get_settings()

    if not settings.enable_entity_extraction:
        logger.error("Entity extraction is disabled. Set RAG_ENABLE_ENTITY_EXTRACTION=true")
        return 1

    repo = IngestionRepository(settings)
    qwen_client = QwenClient(settings)

    logger.info("Fetching documents from kb_documents...")
    docs = list(
        get_documents(
            repo,
            category=args.category,
            doc_ids=args.doc_ids,
            limit=args.limit,
        )
    )
    logger.info("Found %d document(s)", len(docs))

    if not docs:
        logger.info("Nothing to process. Exiting.")
        return 0

    if not args.force:
        logger.info("Checking for documents that already have entities...")
        existing = get_docs_with_entities(repo)
        before = len(docs)
        docs = [d for d in docs if d.doc_id not in existing]
        skipped = before - len(docs)
        if skipped:
            logger.info(
                "Skipping %d docs that already have entities (use --force to re-extract)", skipped
            )

    if not docs:
        logger.info("All documents already have entities. Nothing to do.")
        return 0

    logger.info("Will process %d document(s)", len(docs))

    if args.dry_run:
        for i, doc in enumerate(docs, 1):
            logger.info(
                "  [%d] %s | %s | %s",
                i,
                doc.doc_id,
                doc.category,
                doc.title[:60],
            )
        logger.info("Dry run — no extraction performed.")
        return 0

    # Process documents
    t_start = time.monotonic()
    total_entities = 0
    total_relations = 0
    succeeded = 0
    failed = 0

    if args.concurrency <= 1:
        # Sequential processing
        for i, doc in enumerate(docs, 1):
            try:
                ec, rc, _ = process_single_doc(doc, repo, qwen_client, settings, i, len(docs))
                total_entities += ec
                total_relations += rc
                succeeded += 1
            except Exception:
                logger.error(
                    "[%d/%d] FAILED doc %s: %s",
                    i,
                    len(docs),
                    doc.doc_id,
                    doc.title[:60],
                    exc_info=True,
                )
                failed += 1
    else:
        # Parallel processing with thread pool
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    process_single_doc,
                    doc,
                    repo,
                    qwen_client,
                    settings,
                    i,
                    len(docs),
                ): doc
                for i, doc in enumerate(docs, 1)
            }

            for future in as_completed(futures):
                doc = futures[future]
                try:
                    ec, rc, _ = future.result()
                    total_entities += ec
                    total_relations += rc
                    succeeded += 1
                except Exception:
                    logger.error(
                        "FAILED doc %s: %s",
                        doc.doc_id,
                        doc.title[:60],
                        exc_info=True,
                    )
                    failed += 1

    elapsed = time.monotonic() - t_start

    logger.info("=" * 60)
    logger.info("Entity extraction complete!")
    logger.info("  Documents: %d succeeded, %d failed, %d total", succeeded, failed, len(docs))
    logger.info("  Entities:  %d total", total_entities)
    logger.info("  Relations: %d total", total_relations)
    logger.info("  Time:      %.1fs (%.1fs/doc avg)", elapsed, elapsed / max(len(docs), 1))
    logger.info("=" * 60)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
