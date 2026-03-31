#!/usr/bin/env python3
"""Diagnostic: run entity extraction on sample chunks and report relation yield.

Picks N chunks from pgvector (including known orphan entities like Rushing Back),
runs extraction with INFO logging to show parse paths and drop rates.

Usage:
    cd apps/rag-service
    uv run python scripts/debug_relation_extraction.py [--chunks N]
"""

from __future__ import annotations

import argparse
import logging
import sys

import sqlalchemy

sys.path.insert(0, ".")

from app.config import Settings
from app.entity_extraction import EntityExtractor
from app.entity_vector_store import EntityVectorStore
from app.qwen_client import QwenClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=5, help="Number of sample chunks")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    settings = Settings()
    store = EntityVectorStore(settings)
    engine = store._get_engine()
    qwen = QwenClient(settings)
    extractor = EntityExtractor(qwen, gleaning_rounds=0)

    target_chunk = "8eaa705e-4ee9-4ac1-a105-6ddf3082a7c9_chunk_0"

    with engine.connect() as conn:
        chunks_data: list[tuple[str, str, str]] = []

        target_result = conn.execute(
            sqlalchemy.text(
                """
            SELECT CAST(chunk_id AS text) AS chunk_id,
                   CAST(doc_id AS text) AS doc_id,
                   chunk_text
            FROM kb_chunks
            WHERE CAST(doc_id AS text) = :doc_id AND chunk_index = 0
            LIMIT 1
        """
            ),
            {"doc_id": "8eaa705e-4ee9-4ac1-a105-6ddf3082a7c9"},
        )
        target_row = target_result.fetchone()
        if target_row:
            chunks_data.append(
                (
                    f"{target_row.doc_id}_chunk_0",
                    target_row.doc_id,
                    target_row.chunk_text,
                )
            )

        other_rows = list(
            conn.execute(
                sqlalchemy.text(
                    """
            SELECT CAST(chunk_id AS text) AS chunk_id,
                   CAST(doc_id AS text) AS doc_id,
                   chunk_index,
                   chunk_text
            FROM kb_chunks
            WHERE CAST(doc_id AS text) != '8eaa705e-4ee9-4ac1-a105-6ddf3082a7c9'
            ORDER BY chunk_id
            LIMIT :limit
        """
                ),
                {"limit": args.chunks - 1},
            )
        )
        for r in other_rows:
            chunks_data.append(
                (
                    f"{r.doc_id}_chunk_{r.chunk_index}",
                    r.doc_id,
                    r.chunk_text,
                )
            )

    print(f"\n{'='*80}")
    print(f"Testing extraction on {len(chunks_data)} chunks")
    print(f"{'='*80}\n")

    total_entities = 0
    total_relations = 0

    for chunk_id, doc_id, text in chunks_data:
        print(f"\n--- Chunk: {chunk_id} (doc: {doc_id}) ---")
        print(f"Text length: {len(text)} chars")
        print(f"Preview: {text[:200]}...")

        try:
            result, trace = extractor.extract(chunk_id, doc_id, text)
            print(f"Result: {len(result.entities)} entities, {len(result.relations)} relations")
            print(f"Trace: {trace.validation_status}")

            if result.entities:
                print("Entities:")
                for e in result.entities:
                    print(f"  [{e.type.value}] {e.name} (id={e.entity_id})")

            if result.relations:
                print("Relations:")
                for r in result.relations:
                    print(f"  {r.source_entity_id} --[{r.type.value}]--> {r.target_entity_id}")
            else:
                print("Relations: NONE")

            total_entities += len(result.entities)
            total_relations += len(result.relations)
        except Exception as exc:
            print(f"FAILED: {exc}")

    print(f"\n{'='*80}")
    print(f"TOTALS: {total_entities} entities, {total_relations} relations")
    print(f"Relation yield: {total_relations/max(total_entities,1)*100:.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
