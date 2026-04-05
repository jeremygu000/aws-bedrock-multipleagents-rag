#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Noise patterns (same as entity_extraction.py / analyze_orphans.py)
# ---------------------------------------------------------------------------
PRONOUN_PATTERN = re.compile(
    r"^(i|me|my|we|us|our|you|your|he|him|his|she|her|they|them|their|it|its|"
    r"this|that|these|those|who|whom|which|what|where|when|how|why)$",
    re.IGNORECASE,
)
ARTICLE_PATTERN = re.compile(r"^(a|an|the|some|any|all|each|every|no|none)$", re.IGNORECASE)
GENERIC_MUSIC_TERMS: frozenset[str] = frozenset({
    "musical works", "musical work", "music", "song", "songs", "track", "tracks",
    "album", "albums", "record", "records", "recording", "recordings",
    "writer", "writers", "artist", "artists", "performer", "performers",
    "member", "members", "publisher", "publishers", "speaker", "speakers",
    "tour", "tours", "concert", "concerts", "festival", "festivals",
    "dance", "performance", "event", "events", "session", "sessions",
    "workshop", "workshops", "panel", "panels", "presentation", "presentations",
    "literary works", "literary work", "works", "work",
})
TIME_PATTERN = re.compile(
    r"^\d{1,2}:\d{2}\s*(am|pm|AM|PM)?(\s*-\s*\d{1,2}:\d{2}\s*(am|pm|AM|PM)?)?$"
)
PURE_NUMBER = re.compile(r"^\d+(\.\d+)?%?$")
ORDINAL_PATTERN = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)


def _is_noise(name: str) -> str | None:
    s = name.strip()
    if not s:
        return "empty"
    if len(s) == 1:
        return "single_char"
    if PRONOUN_PATTERN.match(s):
        return "pronoun"
    if ARTICLE_PATTERN.match(s):
        return "article"
    if s.lower() in GENERIC_MUSIC_TERMS:
        return "generic_term"
    if TIME_PATTERN.match(s):
        return "time_fragment"
    if PURE_NUMBER.match(s):
        return "pure_number"
    if ORDINAL_PATTERN.match(s):
        return "ordinal"
    return None


def _resolve_neo4j_password() -> str:
    password = os.environ.get("RAG_NEO4J_PASSWORD", "")
    if password:
        return password
    secret_arn = os.environ.get("RAG_NEO4J_PASSWORD_SECRET_ARN", "")
    if not secret_arn:
        return ""
    client = boto3.client(
        "secretsmanager", region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2")
    )
    resp = client.get_secret_value(SecretId=secret_arn)
    secret = json.loads(resp["SecretString"])
    return secret.get("password", secret.get("neo4j", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up noise from Neo4j graph")
    parser.add_argument(
        "--execute", action="store_true", help="Actually delete (default is dry-run)"
    )
    args = parser.parse_args()

    dry_run = not args.execute

    uri = os.environ.get("RAG_NEO4J_URI", "")
    user = os.environ.get("RAG_NEO4J_USERNAME", "neo4j")
    password = _resolve_neo4j_password()

    if not uri or not password:
        print(f"ERROR: uri={bool(uri)}, password={bool(password)}")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    mode = "DRY-RUN" if dry_run else "EXECUTE"
    print(f"Mode: {mode}")
    print(f"Neo4j: {uri}\n")

    with driver.session(database="neo4j") as session:
        # ---------------------------------------------------------------
        # 1. Noise entities that are ALSO orphans (degree 0)
        # ---------------------------------------------------------------
        print("=" * 60)
        print("  Phase 1: Noise entities (orphans only)")
        print("=" * 60)

        all_entities = session.run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r]-()
            WITH e, id(e) AS neo4j_id, count(r) AS degree
            RETURN neo4j_id, e.name AS name, degree
            """
        ).data()

        noise_orphan_neo4j_ids: list[int] = []
        noise_connected_count = 0
        noise_by_category: dict[str, list[str]] = {}
        for row in all_entities:
            cat = _is_noise(row["name"] or "")
            if cat:
                if row["degree"] == 0:
                    noise_orphan_neo4j_ids.append(row["neo4j_id"])
                    noise_by_category.setdefault(cat, []).append(row["name"])
                else:
                    noise_connected_count += 1

        total_noise = len(noise_orphan_neo4j_ids) + noise_connected_count
        print(f"\nTotal entities in graph: {len(all_entities)}")
        print(f"Total noise entities:    {total_noise}")
        print(f"  Orphaned (will delete): {len(noise_orphan_neo4j_ids)}")
        print(f"  Connected (keeping):    {noise_connected_count}")
        for cat, names in sorted(noise_by_category.items(), key=lambda x: -len(x[1])):
            sample = names[:5]
            print(f"  {cat}: {len(names)}  (e.g. {sample})")

        if noise_orphan_neo4j_ids:
            if not dry_run:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE id(e) IN $ids
                    DELETE e
                    """,
                    ids=noise_orphan_neo4j_ids,
                ).consume()
                print(f"  ✅ Deleted {result.counters.nodes_deleted} orphaned noise entities")
            else:
                print("  (dry-run — no changes made)")

        # ---------------------------------------------------------------
        # 2. Self-referencing relations
        # ---------------------------------------------------------------
        print(f"\n{'=' * 60}")
        print("  Phase 2: Self-referencing relations")
        print("=" * 60)

        self_refs = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.entity_id = b.entity_id
            RETURN a.name AS name, type(r) AS rel_type, r.relation_id AS rid
            """
        ).data()

        print(f"\nSelf-referencing relations found: {len(self_refs)}")
        for sr in self_refs[:10]:
            print(f"  {sr['name']} --[{sr['rel_type']}]--> (self)")
        if len(self_refs) > 10:
            print(f"  ... and {len(self_refs) - 10} more")

        if self_refs and not dry_run:
            result = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.entity_id = b.entity_id
                DELETE r
                RETURN count(r) AS deleted
                """
            ).consume()
            print(f"  ✅ Deleted {result.counters.relationships_deleted} self-referencing relations")
        elif self_refs:
            print("  (dry-run — no changes made)")

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------
        print(f"\n{'=' * 60}")
        print("  Summary")
        print("=" * 60)
        if dry_run:
            print(f"\n  WOULD delete: {len(noise_orphan_neo4j_ids)} orphaned noise entities")
            print(f"  WOULD delete: {len(self_refs)} self-referencing relations")
            print("\n  Run with --execute to apply changes.")
        else:
            stats = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WITH e, count(r) AS degree
                RETURN count(e) AS total_entities,
                       sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS orphans
                """
            ).single()
            total = stats["total_entities"]
            orphans = stats["orphans"]
            rate = (orphans / total * 100) if total else 0
            print("\n  Post-cleanup stats:")
            print(f"    Entities:    {total}")
            print(f"    Orphans:     {orphans} ({rate:.1f}%)")
            print(f"    Connected:   {total - orphans} ({100 - rate:.1f}%)")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
