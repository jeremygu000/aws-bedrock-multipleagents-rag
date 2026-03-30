"""Backfill pgvector kb_entities & kb_relations from Neo4j.

Reads all entities and relations from Neo4j (the healthy source of truth),
generates deterministic entity_ids via SHA-256 hash of (name.lower(), type),
calls Qwen embedding API for each entity/relation, then TRUNCATES and
batch-upserts to pgvector.

This script exists because the original ingestion pipeline used the LLM's
per-chunk local entity_id (e.g. "e1", "e2") as the pgvector primary key,
causing massive cross-document overwrites (5,170 entities → 84 survivors).
Neo4j was unaffected because it MERGEs on (name, type).

Usage:
    cd apps/rag-service && source ../../.envrc
    export QWEN_API_KEY=... LLM_MODEL=qwen-plus
    python -m scripts.backfill_pgvector_from_neo4j [OPTIONS]

Options:
    --dry-run       Show counts only, don't write to pgvector
    --skip-truncate Don't truncate before insert (for incremental runs)
    --batch-size N  Embedding batch size (default: 10, Qwen API max)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from typing import Any

from neo4j import GraphDatabase
from sqlalchemy import text

from app.config import Settings, get_settings
from app.entity_vector_store import EntityVectorStore
from app.qwen_client import QwenClient
from app.secrets import resolve_db_password, resolve_neo4j_password

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.backfill_pgvector")


def make_entity_id(name: str, entity_type: str) -> str:
    """Deterministic entity_id = first 16 hex chars of SHA-256(name.lower()::type)."""
    key = f"{name.lower()}::{entity_type}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def make_relation_id(source_entity_id: str, rel_type: str, target_entity_id: str) -> str:
    """Deterministic relation_id from source_entity_id__rel_type__target_entity_id."""
    return f"{source_entity_id}__{rel_type}__{target_entity_id}"


def read_all_entities(driver: Any, database: str) -> list[dict]:
    """Read all Entity nodes from Neo4j.

    Returns list of dicts with keys matching ExtractedEntity fields.
    """
    cypher = """
    MATCH (e:Entity)
    RETURN e.name AS name,
           e.type AS type,
           e.entity_id AS entity_id,
           e.canonical_key AS canonical_key,
           e.description AS description,
           e.aliases AS aliases,
           e.confidence AS confidence,
           e.source_chunk_ids AS source_chunk_ids
    """
    with driver.session(database=database) as session:
        result = session.run(cypher)
        entities = []
        for record in result:
            entities.append(
                {
                    "name": record["name"] or "",
                    "type": record["type"] or "Work",
                    "old_entity_id": record["entity_id"] or "",
                    "canonical_key": record["canonical_key"] or None,
                    "description": record["description"] or "",
                    "aliases": list(record["aliases"] or []),
                    "confidence": float(record["confidence"] or 0.0),
                    "source_chunk_ids": list(record["source_chunk_ids"] or []),
                }
            )
        return entities


def read_all_relations(driver: Any, database: str) -> list[dict]:
    """Read all RELATES_TO relations from Neo4j.

    Returns list of dicts with source/target entity names and types
    for resolving entity_ids after hashing.
    """
    cypher = """
    MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
    RETURN src.name AS source_name,
           src.type AS source_type,
           tgt.name AS target_name,
           tgt.type AS target_type,
           r.rel_type AS rel_type,
           r.evidence AS evidence,
           r.confidence AS confidence,
           r.weight AS weight,
           r.source_chunk_ids AS source_chunk_ids
    """
    with driver.session(database=database) as session:
        result = session.run(cypher)
        relations = []
        for record in result:
            relations.append(
                {
                    "source_name": record["source_name"] or "",
                    "source_type": record["source_type"] or "Work",
                    "target_name": record["target_name"] or "",
                    "target_type": record["target_type"] or "Work",
                    "rel_type": record["rel_type"] or "RELATES_TO",
                    "evidence": record["evidence"] or "",
                    "confidence": float(record["confidence"] or 0.0),
                    "weight": float(record["weight"] or 1.0),
                    "source_chunk_ids": list(record["source_chunk_ids"] or []),
                }
            )
        return relations


def generate_embeddings(
    qwen: QwenClient,
    texts: list[str],
    batch_size: int = 10,
    label: str = "items",
    max_retries: int = 3,
) -> list[list[float]]:
    """Generate embeddings in batches, respecting Qwen API limits."""
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(
            "Embedding %s batch %d/%d (%d items)...",
            label,
            batch_num,
            total_batches,
            len(batch),
        )

        for attempt in range(1, max_retries + 1):
            try:
                result = qwen.embedding(batch)
                if isinstance(result[0], float):
                    all_embeddings.append(result)
                else:
                    all_embeddings.extend(result)
                break
            except Exception as exc:
                if attempt < max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "Embedding %s batch %d failed (attempt %d/%d): %s. " "Retrying in %ds...",
                        label,
                        batch_num,
                        attempt,
                        max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Embedding %s batch %d failed after %d attempts: %s",
                        label,
                        batch_num,
                        max_retries,
                        exc,
                    )
                    raise

        if i + batch_size < total:
            time.sleep(0.3)

    return all_embeddings


def truncate_entity_tables(settings: Settings) -> None:
    """TRUNCATE kb_relations then kb_entities (FK order)."""
    from sqlalchemy import create_engine

    password = resolve_db_password(settings)
    url = settings.build_db_dsn(password)
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE kb_entities CASCADE"))
    logger.info("Truncated kb_entities (CASCADE → kb_relations)")
    engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pgvector entities/relations from Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only")
    parser.add_argument(
        "--skip-truncate",
        action="store_true",
        help="Don't truncate before insert",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Embedding batch size (default: 10)",
    )
    args = parser.parse_args()

    settings = get_settings()

    if not settings.enable_neo4j:
        logger.error("Neo4j is disabled (RAG_ENABLE_NEO4J=false). Cannot backfill.")
        sys.exit(1)

    neo4j_password = resolve_neo4j_password(settings)
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, neo4j_password),
    )
    logger.info("Connected to Neo4j at %s", settings.neo4j_uri)

    try:
        logger.info("Reading entities from Neo4j...")
        neo4j_entities = read_all_entities(driver, settings.neo4j_database)
        logger.info("Read %d entities from Neo4j", len(neo4j_entities))

        logger.info("Reading relations from Neo4j...")
        neo4j_relations = read_all_relations(driver, settings.neo4j_database)
        logger.info("Read %d relations from Neo4j", len(neo4j_relations))

    finally:
        driver.close()
        logger.info("Neo4j driver closed")

    if not neo4j_entities:
        logger.warning("No entities found in Neo4j. Nothing to backfill.")
        return

    logger.info("Generating deterministic entity_ids...")
    entity_id_map: dict[tuple[str, str], str] = {}
    for entity in neo4j_entities:
        eid = make_entity_id(entity["name"], entity["type"])
        entity["entity_id"] = eid
        entity_id_map[(entity["name"].lower(), entity["type"])] = eid

    unique_ids = set(e["entity_id"] for e in neo4j_entities)
    if len(unique_ids) != len(neo4j_entities):
        logger.warning(
            "Hash collision detected! %d unique IDs for %d entities",
            len(unique_ids),
            len(neo4j_entities),
        )
    else:
        logger.info("All %d entity_ids are unique (no hash collisions)", len(unique_ids))

    logger.info("Generating deterministic relation_ids...")
    skipped_relations = 0
    valid_relations: list[dict] = []
    for rel in neo4j_relations:
        src_key = (rel["source_name"].lower(), rel["source_type"])
        tgt_key = (rel["target_name"].lower(), rel["target_type"])
        src_eid = entity_id_map.get(src_key)
        tgt_eid = entity_id_map.get(tgt_key)
        if not src_eid or not tgt_eid:
            skipped_relations += 1
            logger.debug(
                "Skipping relation: source=%s/%s target=%s/%s (entity not found)",
                rel["source_name"],
                rel["source_type"],
                rel["target_name"],
                rel["target_type"],
            )
            continue
        rel["source_entity_id"] = src_eid
        rel["target_entity_id"] = tgt_eid
        rel["relation_id"] = make_relation_id(src_eid, rel["rel_type"], tgt_eid)
        valid_relations.append(rel)

    if skipped_relations:
        logger.warning("Skipped %d relations (missing entity references)", skipped_relations)
    logger.info("Valid relations: %d", len(valid_relations))

    print(f"\n{'='*60}")
    print("Backfill Summary")
    print(f"{'='*60}")
    print(f"  Entities from Neo4j:    {len(neo4j_entities)}")
    print(f"  Relations from Neo4j:   {len(neo4j_relations)}")
    print(f"  Valid relations:        {len(valid_relations)}")
    print(f"  Skipped relations:      {skipped_relations}")
    print(f"  Unique entity_ids:      {len(unique_ids)}")
    print(f"  Embedding batch size:   {args.batch_size}")
    print(f"{'='*60}\n")

    if args.dry_run:
        logger.info("Dry run — skipping embedding generation and pgvector writes.")
        print("Sample entities (first 5):")
        for e in neo4j_entities[:5]:
            print(f"  {e['entity_id']}: {e['name']} ({e['type']})")
        print("\nSample relations (first 5):")
        for r in valid_relations[:5]:
            print(
                f"  {r['relation_id']}: {r['source_name']} --[{r['rel_type']}]--> {r['target_name']}"
            )
        return

    qwen = QwenClient(settings)

    entity_texts = [f"{e['name']} {e['type']} {e['description']}" for e in neo4j_entities]
    logger.info("Generating embeddings for %d entities...", len(entity_texts))
    t0 = time.time()
    entity_embeddings = generate_embeddings(
        qwen, entity_texts, batch_size=args.batch_size, label="entity"
    )
    logger.info(
        "Entity embeddings done in %.1fs (%d vectors)",
        time.time() - t0,
        len(entity_embeddings),
    )

    assert len(entity_embeddings) == len(
        neo4j_entities
    ), f"Embedding count mismatch: {len(entity_embeddings)} != {len(neo4j_entities)}"

    relation_texts = [f"{r['rel_type']} {r['evidence']}" for r in valid_relations]
    logger.info("Generating embeddings for %d relations...", len(relation_texts))
    t0 = time.time()
    relation_embeddings = generate_embeddings(
        qwen, relation_texts, batch_size=args.batch_size, label="relation"
    )
    logger.info(
        "Relation embeddings done in %.1fs (%d vectors)",
        time.time() - t0,
        len(relation_embeddings),
    )

    assert len(relation_embeddings) == len(
        valid_relations
    ), f"Embedding count mismatch: {len(relation_embeddings)} != {len(valid_relations)}"

    if not args.skip_truncate:
        logger.info("Truncating kb_entities and kb_relations...")
        truncate_entity_tables(settings)
    else:
        logger.info("Skipping truncate (--skip-truncate)")

    vector_store = EntityVectorStore(settings)

    entity_dicts = []
    for idx, entity in enumerate(neo4j_entities):
        entity_dicts.append(
            {
                "entity_id": entity["entity_id"],
                "name": entity["name"],
                "type": entity["type"],
                "description": entity["description"],
                "embedding": entity_embeddings[idx],
                "canonical_key": entity.get("canonical_key"),
                "aliases": entity.get("aliases", []),
                "confidence": entity.get("confidence", 0.0),
                "source_chunk_ids": entity.get("source_chunk_ids", []),
                "metadata": {},
            }
        )

    logger.info("Upserting %d entities to pgvector...", len(entity_dicts))
    t0 = time.time()
    entity_count = vector_store.batch_upsert_entities(entity_dicts)
    logger.info("Entity upsert done in %.1fs (%d upserted)", time.time() - t0, entity_count)

    relation_dicts = []
    for idx, rel in enumerate(valid_relations):
        relation_dicts.append(
            {
                "relation_id": rel["relation_id"],
                "source_entity_id": rel["source_entity_id"],
                "target_entity_id": rel["target_entity_id"],
                "type": rel["rel_type"],
                "evidence": rel["evidence"],
                "embedding": relation_embeddings[idx],
                "confidence": rel.get("confidence", 0.0),
                "weight": rel.get("weight", 1.0),
                "source_chunk_ids": rel.get("source_chunk_ids", []),
                "metadata": {},
            }
        )

    logger.info("Upserting %d relations to pgvector...", len(relation_dicts))
    t0 = time.time()
    relation_count = vector_store.batch_upsert_relations(relation_dicts)
    logger.info(
        "Relation upsert done in %.1fs (%d upserted)",
        time.time() - t0,
        relation_count,
    )

    print(f"\n{'='*60}")
    print("Backfill Complete")
    print(f"{'='*60}")
    print(f"  Entities upserted:  {entity_count}")
    print(f"  Relations upserted: {relation_count}")
    print(f"{'='*60}")
    print("\nRun verify_entity_data.py to confirm counts match Neo4j.")


if __name__ == "__main__":
    main()
