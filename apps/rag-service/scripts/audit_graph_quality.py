#!/usr/bin/env python3
"""Audit Neo4j graph quality after entity extraction.

Reports:
  1. Total entity / relation counts
  2. Orphaned entity rate (entities with zero relations)
  3. Entity type distribution
  4. Relation type distribution
  5. Connectivity stats (avg degree, connected components)
  6. Top-connected entities (hubs)
  7. Per-document entity & relation density
  8. Data quality checks (missing fields, low confidence)

Usage:
    cd apps/rag-service && source ../../.envrc
    python scripts/audit_graph_quality.py
    python scripts/audit_graph_quality.py --top 20          # top-N hubs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
from neo4j import GraphDatabase


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


def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Neo4j graph quality")
    parser.add_argument("--top", type=int, default=15, help="Top-N hubs to show")
    args = parser.parse_args()

    uri = os.environ.get("RAG_NEO4J_URI", "")
    user = os.environ.get("RAG_NEO4J_USERNAME", "neo4j")
    password = _resolve_neo4j_password()

    if not uri or not password:
        print(f"ERROR: uri={bool(uri)}, password={bool(password)}")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database="neo4j") as session:

        # ---------------------------------------------------------------
        # 1. Total counts
        # ---------------------------------------------------------------
        _header("1. TOTAL COUNTS")
        r = session.run("MATCH (e:Entity) RETURN count(e) AS cnt").single()
        total_entities = r["cnt"]

        r = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt").single()
        total_relations = r["cnt"]

        ratio = round(total_relations / total_entities, 2) if total_entities else 0
        print(f"  Entities:  {total_entities:,}")
        print(f"  Relations: {total_relations:,}")
        print(f"  Ratio:     {ratio} relations per entity")

        # ---------------------------------------------------------------
        # 2. Orphaned entities
        # ---------------------------------------------------------------
        _header("2. ORPHANED ENTITIES (no relations)")
        r = session.run(
            "MATCH (e:Entity) WHERE NOT (e)-[:RELATES_TO]-() RETURN count(e) AS cnt"
        ).single()
        orphaned = r["cnt"]
        orphan_pct = round(100 * orphaned / total_entities, 1) if total_entities else 0
        print(f"  Orphaned:  {orphaned:,} / {total_entities:,}  ({orphan_pct}%)")

        connected = total_entities - orphaned
        print(f"  Connected: {connected:,}  ({round(100 - orphan_pct, 1)}%)")

        # ---------------------------------------------------------------
        # 3. Entity type distribution
        # ---------------------------------------------------------------
        _header("3. ENTITY TYPE DISTRIBUTION")
        results = session.run(
            "MATCH (e:Entity) RETURN e.type AS type, count(*) AS cnt ORDER BY cnt DESC"
        )
        rows = list(results)
        for r in rows:
            pct = round(100 * r["cnt"] / total_entities, 1) if total_entities else 0
            print(f"  {r['type']:25s}  {r['cnt']:6,}  ({pct}%)")

        # ---------------------------------------------------------------
        # 4. Relation type distribution
        # ---------------------------------------------------------------
        _header("4. RELATION TYPE DISTRIBUTION")
        results = session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.rel_type AS type, count(*) AS cnt ORDER BY cnt DESC"
        )
        rows = list(results)
        for r in rows:
            pct = round(100 * r["cnt"] / total_relations, 1) if total_relations else 0
            print(f"  {r['type']:30s}  {r['cnt']:6,}  ({pct}%)")

        # ---------------------------------------------------------------
        # 5. Connectivity stats (degree distribution)
        # ---------------------------------------------------------------
        _header("5. CONNECTIVITY STATS")
        results = session.run("""
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]-()
            WITH e, count(r) AS degree
            RETURN
                min(degree) AS min_deg,
                max(degree) AS max_deg,
                avg(degree) AS avg_deg,
                percentileCont(degree, 0.5) AS median_deg,
                percentileCont(degree, 0.9) AS p90_deg,
                percentileCont(degree, 0.99) AS p99_deg,
                stDev(degree) AS std_deg
        """).single()
        print(f"  Min degree:    {results['min_deg']}")
        print(f"  Max degree:    {results['max_deg']}")
        print(f"  Avg degree:    {round(results['avg_deg'], 2)}")
        print(f"  Median degree: {results['median_deg']}")
        print(f"  P90 degree:    {results['p90_deg']}")
        print(f"  P99 degree:    {results['p99_deg']}")
        print(f"  Std dev:       {round(results['std_deg'], 2)}")

        # ---------------------------------------------------------------
        # 6. Top-connected entities (hubs)
        # ---------------------------------------------------------------
        _header(f"6. TOP-{args.top} MOST CONNECTED ENTITIES (hubs)")
        results = session.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]-()
            WITH e, count(r) AS degree
            ORDER BY degree DESC LIMIT $top
            RETURN e.name AS name, e.type AS type, degree
            """,
            top=args.top,
        )
        for r in results:
            print(f"  {r['degree']:4d}  {r['type']:15s}  {r['name']}")

        # ---------------------------------------------------------------
        # 7. Orphan entity types (which types are most orphaned?)
        # ---------------------------------------------------------------
        _header("7. ORPHAN RATE BY ENTITY TYPE")
        results = session.run("""
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]-()
            WITH e.type AS type, count(e) AS total, sum(CASE WHEN r IS NULL THEN 1 ELSE 0 END) AS orphans
            RETURN type, total, orphans,
                   round(100.0 * orphans / total, 1) AS orphan_pct
            ORDER BY orphan_pct DESC
        """)
        rows = list(results)
        print(f"  {'Type':25s}  {'Total':>6s}  {'Orphans':>7s}  {'Orphan%':>7s}")
        print(f"  {'-'*25}  {'-'*6}  {'-'*7}  {'-'*7}")
        for r in rows:
            print(f"  {r['type']:25s}  {r['total']:6,}  {r['orphans']:7,}  {r['orphan_pct']:6.1f}%")

        # ---------------------------------------------------------------
        # 8. Data quality: missing descriptions, low confidence
        # ---------------------------------------------------------------
        _header("8. DATA QUALITY CHECKS")

        r = session.run(
            "MATCH (e:Entity) WHERE e.description IS NULL OR e.description = '' RETURN count(e) AS cnt"
        ).single()
        no_desc = r["cnt"]
        print(f"  Entities missing description: {no_desc:,}")

        r = session.run(
            "MATCH (e:Entity) WHERE e.confidence < 0.5 RETURN count(e) AS cnt"
        ).single()
        low_conf = r["cnt"]
        print(f"  Entities with confidence < 0.5: {low_conf:,}")

        r = session.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.confidence < 0.5 RETURN count(r) AS cnt"
        ).single()
        low_conf_rel = r["cnt"]
        print(f"  Relations with confidence < 0.5: {low_conf_rel:,}")

        r = session.run(
            "MATCH (e:Entity) WHERE size(e.source_chunk_ids) > 1 RETURN count(e) AS cnt"
        ).single()
        multi_source = r["cnt"]
        print(f"  Entities appearing in 2+ chunks: {multi_source:,} (cross-referenced)")

        # ---------------------------------------------------------------
        # 9. Sample relations (sanity check)
        # ---------------------------------------------------------------
        _header("9. SAMPLE RELATIONS (random 15)")
        results = session.run("""
            MATCH (s)-[r:RELATES_TO]->(t)
            WITH s, r, t, rand() AS rnd
            ORDER BY rnd LIMIT 15
            RETURN s.name AS src, s.type AS src_type,
                   r.rel_type AS rel, r.confidence AS conf,
                   t.name AS tgt, t.type AS tgt_type
        """)
        for r in results:
            conf = round(r["conf"], 2) if r["conf"] is not None else "?"
            print(f"  [{r['src_type']:12s}] {r['src']:30s} --{r['rel']:20s}--> [{r['tgt_type']:12s}] {r['tgt']}  conf={conf}")

        # ---------------------------------------------------------------
        # 10. Summary scorecard
        # ---------------------------------------------------------------
        _header("10. SUMMARY SCORECARD")
        print(f"  Total entities:      {total_entities:,}")
        print(f"  Total relations:     {total_relations:,}")
        print(f"  Relation/entity:     {ratio}")
        print(f"  Orphan rate:         {orphan_pct}%", end="")
        if orphan_pct < 20:
            print("  ✅ GOOD")
        elif orphan_pct < 40:
            print("  ⚠️  MODERATE")
        else:
            print("  ❌ HIGH")
        print(f"  Cross-referenced:    {multi_source:,} entities", end="")
        if total_entities and multi_source / total_entities > 0.1:
            print("  ✅ GOOD")
        else:
            print("  ⚠️  LOW")

    driver.close()
    print("\n✅ Audit complete.")


if __name__ == "__main__":
    main()
