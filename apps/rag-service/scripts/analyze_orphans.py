#!/usr/bin/env python3
"""Deep analysis of orphaned entities in Neo4j graph.

Characterizes WHY 68.8% of entities have zero relations by examining:
  1. Name length distribution (orphan vs connected)
  2. Name pattern analysis (pronouns, generic terms, numbers, time fragments)
  3. Orphan rate by entity type
  4. Most common orphan names (frequency)
  5. Single-char and very short entity names
  6. Entity names that look like noise (stopwords, pronouns, articles)
  7. Connected entity examples (what good entities look like)
  8. Chunk-level analysis: chunks with ALL orphans vs mixed

Usage:
    cd apps/rag-service && source ../../.envrc
    uv run python scripts/analyze_orphans.py
    uv run python scripts/analyze_orphans.py --top 30    # more examples
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

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


# Noise patterns — entities that shouldn't have been extracted
PRONOUN_PATTERN = re.compile(
    r"^(i|me|my|we|us|our|you|your|he|him|his|she|her|they|them|their|it|its|"
    r"this|that|these|those|who|whom|which|what|where|when|how|why)$",
    re.IGNORECASE,
)
ARTICLE_PATTERN = re.compile(r"^(a|an|the|some|any|all|each|every|no|none)$", re.IGNORECASE)
GENERIC_MUSIC_TERMS = {
    "musical works", "musical work", "music", "song", "songs", "track", "tracks",
    "album", "albums", "record", "records", "recording", "recordings",
    "writer", "writers", "artist", "artists", "performer", "performers",
    "member", "members", "publisher", "publishers", "speaker", "speakers",
    "tour", "tours", "concert", "concerts", "festival", "festivals",
    "dance", "performance", "event", "events", "session", "sessions",
    "workshop", "workshops", "panel", "panels", "presentation", "presentations",
    "literary works", "literary work", "works", "work",
}
TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}\s*(am|pm|AM|PM)?(\s*-\s*\d{1,2}:\d{2}\s*(am|pm|AM|PM)?)?$")
PURE_NUMBER = re.compile(r"^\d+(\.\d+)?%?$")
ORDINAL_PATTERN = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)


def classify_noise(name: str) -> str | None:
    """Classify an entity name as noise, returning the noise category or None."""
    stripped = name.strip()
    if not stripped:
        return "empty"
    if len(stripped) == 1:
        return "single_char"
    if PRONOUN_PATTERN.match(stripped):
        return "pronoun"
    if ARTICLE_PATTERN.match(stripped):
        return "article"
    if stripped.lower() in GENERIC_MUSIC_TERMS:
        return "generic_term"
    if TIME_PATTERN.match(stripped):
        return "time_fragment"
    if PURE_NUMBER.match(stripped):
        return "pure_number"
    if ORDINAL_PATTERN.match(stripped):
        return "ordinal"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze orphaned entities in Neo4j")
    parser.add_argument("--top", type=int, default=20, help="Top-N examples to show")
    args = parser.parse_args()

    uri = os.environ.get("RAG_NEO4J_URI", "")
    user = os.environ.get("RAG_NEO4J_USERNAME", "neo4j")
    password = _resolve_neo4j_password()

    if not uri or not password:
        print(f"ERROR: uri={bool(uri)}, password={bool(password)}")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database="neo4j") as session:
        # Fetch ALL entities with their degree
        print("Fetching all entities with degree info...")
        results = session.run("""
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]-()
            WITH e, count(r) AS degree
            RETURN e.name AS name, e.type AS type, degree,
                   size(e.source_chunk_ids) AS chunk_count
        """)
        entities = [dict(r) for r in results]
        print(f"  Loaded {len(entities):,} entities\n")

        orphans = [e for e in entities if e["degree"] == 0]
        connected = [e for e in entities if e["degree"] > 0]

        # ---------------------------------------------------------------
        # 1. Name length distribution
        # ---------------------------------------------------------------
        _header("1. NAME LENGTH DISTRIBUTION")
        orphan_lens = [len(e["name"]) for e in orphans]
        conn_lens = [len(e["name"]) for e in connected]

        def _stats(vals: list[int]) -> dict:
            if not vals:
                return {"min": 0, "max": 0, "avg": 0, "median": 0}
            s = sorted(vals)
            return {
                "min": s[0],
                "max": s[-1],
                "avg": round(sum(s) / len(s), 1),
                "median": s[len(s) // 2],
            }

        o_stats = _stats(orphan_lens)
        c_stats = _stats(conn_lens)

        print(f"  {'':20s}  {'Orphans':>10s}  {'Connected':>10s}")
        print(f"  {'':20s}  {'-------':>10s}  {'---------':>10s}")
        print(f"  {'Count':20s}  {len(orphans):>10,}  {len(connected):>10,}")
        print(f"  {'Avg name length':20s}  {o_stats['avg']:>10}  {c_stats['avg']:>10}")
        print(f"  {'Median name length':20s}  {o_stats['median']:>10}  {c_stats['median']:>10}")
        print(f"  {'Min name length':20s}  {o_stats['min']:>10}  {c_stats['min']:>10}")
        print(f"  {'Max name length':20s}  {o_stats['max']:>10}  {c_stats['max']:>10}")

        # Buckets
        def _bucket(length: int) -> str:
            if length <= 2:
                return "1-2 chars"
            if length <= 5:
                return "3-5 chars"
            if length <= 10:
                return "6-10 chars"
            if length <= 20:
                return "11-20 chars"
            if length <= 50:
                return "21-50 chars"
            return "51+ chars"

        orphan_buckets = Counter(_bucket(x) for x in orphan_lens)
        conn_buckets = Counter(_bucket(x) for x in conn_lens)
        all_buckets = ["1-2 chars", "3-5 chars", "6-10 chars", "11-20 chars", "21-50 chars", "51+ chars"]

        print(f"\n  {'Bucket':20s}  {'Orphans':>10s}  {'Connected':>10s}  {'Orphan%':>8s}")
        print(f"  {'------':20s}  {'-------':>10s}  {'---------':>10s}  {'-------':>8s}")
        for b in all_buckets:
            o = orphan_buckets.get(b, 0)
            c = conn_buckets.get(b, 0)
            total = o + c
            pct = round(100 * o / total, 1) if total else 0
            print(f"  {b:20s}  {o:>10,}  {c:>10,}  {pct:>7.1f}%")

        # ---------------------------------------------------------------
        # 2. Noise classification
        # ---------------------------------------------------------------
        _header("2. NOISE CLASSIFICATION")
        orphan_noise = Counter()
        connected_noise = Counter()

        for e in orphans:
            cat = classify_noise(e["name"])
            if cat:
                orphan_noise[cat] += 1

        for e in connected:
            cat = classify_noise(e["name"])
            if cat:
                connected_noise[cat] += 1

        total_orphan_noise = sum(orphan_noise.values())
        total_conn_noise = sum(connected_noise.values())

        print(f"  Orphans with noise names:    {total_orphan_noise:,} / {len(orphans):,} ({round(100 * total_orphan_noise / len(orphans), 1) if orphans else 0}%)")
        print(f"  Connected with noise names:  {total_conn_noise:,} / {len(connected):,} ({round(100 * total_conn_noise / len(connected), 1) if connected else 0}%)")
        print()

        all_cats = sorted(set(list(orphan_noise.keys()) + list(connected_noise.keys())))
        print(f"  {'Category':20s}  {'Orphans':>10s}  {'Connected':>10s}")
        print(f"  {'--------':20s}  {'-------':>10s}  {'---------':>10s}")
        for cat in all_cats:
            print(f"  {cat:20s}  {orphan_noise.get(cat, 0):>10,}  {connected_noise.get(cat, 0):>10,}")

        # ---------------------------------------------------------------
        # 3. Most common orphan names (duplicates across chunks)
        # ---------------------------------------------------------------
        _header(f"3. TOP-{args.top} MOST COMMON ORPHAN NAMES")
        orphan_name_count = Counter(e["name"].lower() for e in orphans)
        print(f"  {'Count':>6s}  {'Type':15s}  Name")
        print(f"  {'-----':>6s}  {'----':15s}  ----")
        for name, count in orphan_name_count.most_common(args.top):
            # Find first entity with this name for type info
            etype = next(e["type"] for e in orphans if e["name"].lower() == name)
            print(f"  {count:>6,}  {etype:15s}  {name}")

        # ---------------------------------------------------------------
        # 4. Orphan rate by entity type (with noise breakdown)
        # ---------------------------------------------------------------
        _header("4. ORPHAN RATE BY TYPE (with noise)")
        type_stats: dict[str, dict] = {}
        for e in entities:
            t = e["type"]
            if t not in type_stats:
                type_stats[t] = {"total": 0, "orphans": 0, "noise_orphans": 0}
            type_stats[t]["total"] += 1
            if e["degree"] == 0:
                type_stats[t]["orphans"] += 1
                if classify_noise(e["name"]):
                    type_stats[t]["noise_orphans"] += 1

        print(f"  {'Type':15s}  {'Total':>7s}  {'Orphans':>8s}  {'Noise':>7s}  {'Real':>7s}  {'Real%':>7s}")
        print(f"  {'-'*15}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
        for t in sorted(type_stats, key=lambda x: type_stats[x]["orphans"], reverse=True):
            s = type_stats[t]
            real_orphans = s["orphans"] - s["noise_orphans"]
            real_pct = round(100 * real_orphans / s["total"], 1) if s["total"] else 0
            print(f"  {t:15s}  {s['total']:>7,}  {s['orphans']:>8,}  {s['noise_orphans']:>7,}  {real_orphans:>7,}  {real_pct:>6.1f}%")

        # ---------------------------------------------------------------
        # 5. Sample orphan entities by type
        # ---------------------------------------------------------------
        _header(f"5. SAMPLE ORPHAN ENTITIES (first {args.top} per type)")
        orphans_by_type: dict[str, list] = {}
        for e in orphans:
            t = e["type"]
            if t not in orphans_by_type:
                orphans_by_type[t] = []
            orphans_by_type[t].append(e)

        for t in sorted(orphans_by_type):
            examples = orphans_by_type[t][:args.top]
            noise_count = sum(1 for e in orphans_by_type[t] if classify_noise(e["name"]))
            print(f"\n  {t} ({len(orphans_by_type[t]):,} orphans, {noise_count:,} noise):")
            for e in examples:
                tag = classify_noise(e["name"])
                tag_str = f" [{tag}]" if tag else ""
                print(f"    - \"{e['name']}\"{tag_str}")

        # ---------------------------------------------------------------
        # 6. Sample connected entities (what good looks like)
        # ---------------------------------------------------------------
        _header(f"6. SAMPLE CONNECTED ENTITIES (top {args.top} by degree)")
        connected_sorted = sorted(connected, key=lambda e: e["degree"], reverse=True)
        print(f"  {'Degree':>6s}  {'Type':15s}  Name")
        print(f"  {'------':>6s}  {'----':15s}  ----")
        for e in connected_sorted[:args.top]:
            print(f"  {e['degree']:>6,}  {e['type']:15s}  {e['name']}")

        # ---------------------------------------------------------------
        # 7. Cross-chunk appearance (orphans in 1 chunk vs multi-chunk)
        # ---------------------------------------------------------------
        _header("7. CHUNK APPEARANCE (orphan vs connected)")
        orphan_single_chunk = sum(1 for e in orphans if e["chunk_count"] <= 1)
        orphan_multi_chunk = sum(1 for e in orphans if e["chunk_count"] > 1)
        conn_single_chunk = sum(1 for e in connected if e["chunk_count"] <= 1)
        conn_multi_chunk = sum(1 for e in connected if e["chunk_count"] > 1)

        print(f"  {'':20s}  {'Orphans':>10s}  {'Connected':>10s}")
        print(f"  {'':20s}  {'-------':>10s}  {'---------':>10s}")
        print(f"  {'Single chunk':20s}  {orphan_single_chunk:>10,}  {conn_single_chunk:>10,}")
        print(f"  {'Multi chunk (2+)':20s}  {orphan_multi_chunk:>10,}  {conn_multi_chunk:>10,}")

        # ---------------------------------------------------------------
        # 8. Self-referencing relations
        # ---------------------------------------------------------------
        _header("8. SELF-REFERENCING RELATIONS")
        results = session.run("""
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE s = t
            RETURN s.name AS name, s.type AS type, r.rel_type AS rel, count(*) AS cnt
            ORDER BY cnt DESC
            LIMIT 20
        """)
        self_refs = list(results)
        print(f"  Total self-referencing relations: {sum(r['cnt'] for r in self_refs)}")
        for r in self_refs:
            print(f"    {r['cnt']:>3}x  [{r['type']:12s}] \"{r['name']}\" --{r['rel']}--> self")

        # ---------------------------------------------------------------
        # 9. Summary & recommendations
        # ---------------------------------------------------------------
        _header("9. SUMMARY & RECOMMENDATIONS")
        total_noise = total_orphan_noise + total_conn_noise
        print(f"  Total entities:           {len(entities):,}")
        print(f"  Orphaned entities:        {len(orphans):,} ({round(100 * len(orphans) / len(entities), 1)}%)")
        print(f"  Noise entities (total):   {total_noise:,} ({round(100 * total_noise / len(entities), 1)}%)")
        print(f"  Noise orphans:            {total_orphan_noise:,} (can be safely filtered)")
        print(f"  Real orphans (non-noise): {len(orphans) - total_orphan_noise:,}")
        print(f"  Self-ref relations:       {sum(r['cnt'] for r in self_refs):,}")
        print()

        adjusted_orphan_rate = round(
            100 * (len(orphans) - total_orphan_noise) / (len(entities) - total_noise), 1
        ) if (len(entities) - total_noise) else 0
        print(f"  Adjusted orphan rate (excluding noise): {adjusted_orphan_rate}%")
        print()
        print("  Recommendations:")
        print("    1. Filter noise entities in extraction prompt (pronouns, articles, generic terms)")
        print("    2. Add minimum name length filter (≥3 chars)")
        print("    3. Remove self-referencing relations in post-processing")
        print("    4. Consider entity deduplication (e.g., 'APRA AMCOS' vs 'APRA' vs 'AMCOS')")

    driver.close()
    print("\n✅ Analysis complete.")


if __name__ == "__main__":
    main()
