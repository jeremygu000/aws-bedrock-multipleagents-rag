"""Verify entity/relation data in pgvector and Neo4j after extraction.

Quick diagnostic script to check counts, dedup quality, and data consistency.

Usage:
    cd apps/rag-service && source ../../.envrc && uv run python scripts/verify_entity_data.py
"""

from __future__ import annotations

import logging
import sys

from sqlalchemy import text

from app.config import get_settings
from app.graph_repository import Neo4jRepository
from app.ingestion_repository import IngestionRepository
from app.secrets import resolve_neo4j_password

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify_entity_data")


def verify_pgvector(settings) -> None:
    """Check entity/relation counts and data quality in pgvector."""
    repo = IngestionRepository(settings)
    engine = repo._get_engine()

    queries = {
        "Total entities": "SELECT COUNT(*) FROM kb_entities",
        "Total relations": "SELECT COUNT(*) FROM kb_relations",
        "Distinct entity names": "SELECT COUNT(DISTINCT name) FROM kb_entities",
        "Distinct (name, type) pairs": "SELECT COUNT(DISTINCT (name, type)) FROM kb_entities",
        "Distinct canonical_keys (non-null)": (
            "SELECT COUNT(DISTINCT canonical_key) FROM kb_entities WHERE canonical_key IS NOT NULL AND canonical_key != ''"
        ),
        "Entities with embeddings": "SELECT COUNT(*) FROM kb_entities WHERE embedding IS NOT NULL",
        "Relations with embeddings": "SELECT COUNT(*) FROM kb_relations WHERE embedding IS NOT NULL",
        "Distinct source docs (from chunk_ids)": """
            SELECT COUNT(DISTINCT split_part(chunk_id, '_chunk_', 1))
            FROM (SELECT unnest(source_chunk_ids) AS chunk_id FROM kb_entities) sub
        """,
    }

    print("\n" + "=" * 60)
    print("pgvector (kb_entities / kb_relations)")
    print("=" * 60)

    with engine.connect() as conn:
        for label, sql in queries.items():
            try:
                result = conn.execute(text(sql)).scalar()
                print(f"  {label}: {result}")
            except Exception as e:
                print(f"  {label}: ERROR — {e}")

    # Entity type breakdown
    print("\n  --- Entity Type Breakdown ---")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT type, COUNT(*) as cnt FROM kb_entities GROUP BY type ORDER BY cnt DESC")
        ).fetchall()
        for row in rows:
            print(f"    {row[0]}: {row[1]}")

    # Relation type breakdown
    print("\n  --- Relation Type Breakdown ---")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT type, COUNT(*) as cnt FROM kb_relations GROUP BY type ORDER BY cnt DESC")
        ).fetchall()
        for row in rows:
            print(f"    {row[0]}: {row[1]}")

    # Sample entities
    print("\n  --- Sample Entities (first 10) ---")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT entity_id, name, type, canonical_key, confidence, array_length(source_chunk_ids, 1) as chunk_count "
                "FROM kb_entities ORDER BY confidence DESC LIMIT 10"
            )
        ).fetchall()
        for row in rows:
            print(
                f"    [{row[2]}] {row[1]} (id={row[0]}, key={row[3]}, conf={row[4]:.2f}, chunks={row[5] or 0})"
            )

    # Check for suspicious entity_ids (e.g., "e1", "e2" — LLM-generated local IDs)
    print("\n  --- Suspicious entity_id patterns ---")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT entity_id, name, type FROM kb_entities "
                "WHERE entity_id ~ '^e[0-9]+$' OR entity_id ~ '^rule_' "
                "ORDER BY entity_id LIMIT 20"
            )
        ).fetchall()
        if rows:
            print(f"    Found {len(rows)} entities with LLM-style IDs (e1, e2, ...):")
            for row in rows:
                print(f"      {row[0]}: {row[1]} ({row[2]})")
        else:
            print("    None found — entity_ids look properly generated")

    # Check for entity_id collisions (same ID, different name — sign of overwrites)
    print("\n  --- Entity ID uniqueness check ---")
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM kb_entities")).scalar()
        distinct_ids = conn.execute(
            text("SELECT COUNT(DISTINCT entity_id) FROM kb_entities")
        ).scalar()
        distinct_names = conn.execute(
            text("SELECT COUNT(DISTINCT lower(name)) FROM kb_entities")
        ).scalar()
        print(f"    Total rows: {total}")
        print(f"    Distinct entity_ids: {distinct_ids}")
        print(f"    Distinct names (lowercase): {distinct_names}")
        if total == distinct_ids:
            print("    ✅ No entity_id collisions")
        else:
            print(f"    ⚠️  {total - distinct_ids} duplicate entity_ids found!")


def verify_neo4j(settings) -> None:
    """Check entity/relation counts in Neo4j."""
    if not settings.enable_neo4j:
        print("\n" + "=" * 60)
        print("Neo4j — DISABLED (RAG_ENABLE_NEO4J=false)")
        print("=" * 60)
        return

    try:
        password = resolve_neo4j_password(settings)
    except Exception as e:
        print(f"\n  Neo4j password resolution failed: {e}")
        return

    try:
        repo = Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=password,
            database=settings.neo4j_database,
        )
    except Exception as e:
        print(f"\n  Neo4j connection failed: {e}")
        return

    print("\n" + "=" * 60)
    print(f"Neo4j ({settings.neo4j_uri})")
    print("=" * 60)

    try:
        if not repo.health_check():
            print("  ❌ Health check failed")
            repo.close()
            return
        print("  ✅ Connected")
    except Exception as e:
        print(f"  ❌ Health check error: {e}")
        repo.close()
        return

    queries = {
        "Total Entity nodes": "MATCH (e:Entity) RETURN count(e) AS cnt",
        "Total relationships": "MATCH ()-[r]->() RETURN count(r) AS cnt",
        "Distinct (name, type) pairs": (
            "MATCH (e:Entity) RETURN count(DISTINCT [e.name, e.type]) AS cnt"
        ),
    }

    driver = repo._get_driver()

    with driver.session(database=settings.neo4j_database) as session:
        for label, cypher in queries.items():
            try:
                result = session.run(cypher).single()
                print(f"  {label}: {result['cnt'] if result else 'N/A'}")
            except Exception as e:
                print(f"  {label}: ERROR — {e}")

        # Type breakdown
        print("\n  --- Entity Type Breakdown ---")
        try:
            results = session.run(
                "MATCH (e:Entity) RETURN e.type AS type, count(e) AS cnt ORDER BY cnt DESC"
            ).data()
            for row in results:
                print(f"    {row['type']}: {row['cnt']}")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Relationship type breakdown
        print("\n  --- Relationship Type Breakdown ---")
        try:
            results = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC"
            ).data()
            for row in results:
                print(f"    {row['type']}: {row['cnt']}")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Sample entities
        print("\n  --- Sample Entities (first 10, highest confidence) ---")
        try:
            results = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, "
                "e.entity_id AS id, e.canonical_key AS key, e.confidence AS conf "
                "ORDER BY e.confidence DESC LIMIT 10"
            ).data()
            for row in results:
                conf = row.get("conf", 0) or 0
                print(
                    f"    [{row['type']}] {row['name']} (id={row['id']}, key={row['key']}, conf={conf:.2f})"
                )
        except Exception as e:
            print(f"    ERROR: {e}")

    repo.close()


def main() -> int:
    settings = get_settings()

    verify_pgvector(settings)
    verify_neo4j(settings)

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
