"""Backfill OpenSearch/Elasticsearch index from existing PostgreSQL chunks.

Reads all chunks from kb_chunks, constructs the same document shape that
bulk_index_opensearch() would produce during normal ingestion, and indexes
them into OpenSearch in batches.

Use this after a crawl that ran without RAG_OPENSEARCH_ENDPOINT configured.

Usage:
    cd apps/rag-service && python -m scripts.backfill_opensearch [OPTIONS]

Options:
    --batch-size N   Chunks per bulk request (default: 500)
    --dry-run        Count chunks and show plan, but don't index
    --category CAT   Filter by document category (e.g. pages, news, awards)
    --recreate       Drop and recreate the index before backfilling
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from typing import Any

from opensearchpy import helpers
from sqlalchemy import text

from app.config import get_settings
from app.ingestion_repository import IngestionRepository

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.backfill_opensearch")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

_FETCH_CHUNKS_SQL = """\
SELECT
    c.chunk_id,
    c.doc_id,
    c.chunk_index,
    c.chunk_text,
    c.citation_url,
    c.citation_title,
    c.citation_year,
    c.citation_month,
    c.page_start,
    c.page_end,
    c.section_id,
    c.anchor_id,
    c.metadata
FROM kb_chunks c
JOIN kb_documents d ON d.doc_id = c.doc_id
{where_clause}
ORDER BY c.doc_id, c.chunk_index
"""


def fetch_chunks(
    repo: IngestionRepository,
    category: str | None = None,
) -> list[dict[str, Any]]:

    where_parts: list[str] = []
    params: dict[str, Any] = {}

    if category:
        where_parts.append("d.category = :category")
        params["category"] = category

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    sql = _FETCH_CHUNKS_SQL.format(where_clause=where_clause)

    engine = repo._get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()

    return [row._asdict() for row in rows]


def build_actions(
    chunks: list[dict[str, Any]],
    index_name: str,
) -> list[dict[str, Any]]:

    actions: list[dict[str, Any]] = []
    for chunk in chunks:
        actions.append(
            {
                "_index": index_name,
                "_id": str(uuid.uuid4()),
                "_source": {
                    "chunk_text": chunk["chunk_text"],
                    "doc_id": str(chunk["doc_id"]),
                    "citation_url": chunk["citation_url"],
                    "citation_title": chunk["citation_title"],
                    "citation_year": chunk["citation_year"],
                    "citation_month": chunk["citation_month"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "section_id": chunk["section_id"],
                    "anchor_id": chunk["anchor_id"],
                    "metadata": chunk["metadata"],
                },
            }
        )
    return actions


def recreate_index(repo: IngestionRepository, index_name: str) -> None:

    client = repo._get_opensearch_client()
    if client.indices.exists(index=index_name):
        logger.info("Deleting existing index '%s'", index_name)
        client.indices.delete(index=index_name)
    logger.info("Creating index '%s'", index_name)
    client.indices.create(index=index_name)


def backfill(
    repo: IngestionRepository,
    chunks: list[dict[str, Any]],
    index_name: str,
    batch_size: int,
) -> int:

    client = repo._get_opensearch_client()
    total = len(chunks)
    indexed = 0

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        actions = build_actions(batch, index_name)
        success_count, errors = helpers.bulk(client, actions)
        indexed += success_count
        if errors:
            logger.warning("Batch %d-%d had %d errors", start, start + len(batch), len(errors))
            for err in errors[:3]:
                logger.warning("  %s", err)
        logger.info(
            "Progress: %d / %d indexed (batch %d-%d)",
            indexed,
            total,
            start,
            start + len(batch),
        )

    return indexed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill OpenSearch from PostgreSQL kb_chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Chunks per bulk request (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without indexing",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by document category",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the index before backfilling",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()

    if not settings.opensearch_endpoint.strip():
        logger.error("RAG_OPENSEARCH_ENDPOINT is not configured. Exiting.")
        sys.exit(1)

    index_name = settings.opensearch_index
    logger.info("OpenSearch endpoint: %s", settings.opensearch_endpoint)
    logger.info("Target index: %s", index_name)

    repo = IngestionRepository(settings)

    # Fetch chunks from PG
    logger.info("Fetching chunks from PostgreSQL...")
    t0 = time.time()
    chunks = fetch_chunks(repo, category=args.category)
    elapsed = time.time() - t0
    logger.info("Found %d chunks in %.1fs", len(chunks), elapsed)

    if not chunks:
        logger.info("No chunks to index. Done.")
        return

    if args.dry_run:
        logger.info(
            "[DRY RUN] Would index %d chunks in batches of %d", len(chunks), args.batch_size
        )
        return

    # Recreate index if requested
    if args.recreate:
        recreate_index(repo, index_name)

    # Backfill
    logger.info("Starting backfill (batch_size=%d)...", args.batch_size)
    t0 = time.time()
    indexed = backfill(repo, chunks, index_name, args.batch_size)
    elapsed = time.time() - t0
    logger.info("Done! Indexed %d / %d chunks in %.1fs", indexed, len(chunks), elapsed)


if __name__ == "__main__":
    main()
