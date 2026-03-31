#!/usr/bin/env python3
"""Debug script to investigate Neo4j neighbor expansion for specific entities.

Usage:
    cd apps/rag-service && source ../../.envrc
    python scripts/debug_neo4j_neighbors.py
"""

from __future__ import annotations

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


def main() -> None:
    uri = os.environ.get("RAG_NEO4J_URI", "")
    user = os.environ.get("RAG_NEO4J_USERNAME", "neo4j")
    password = _resolve_neo4j_password()

    if not uri or not password:
        print(f"ERROR: uri={bool(uri)}, password={bool(password)}")
        print("Set RAG_NEO4J_URI and (RAG_NEO4J_PASSWORD or RAG_NEO4J_PASSWORD_SECRET_ARN)")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database="neo4j") as session:
        print("=" * 80)
        print("1. Check if 'Rushing Back' entity exists")
        print("=" * 80)
        result = session.run(
            "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'rushing back' RETURN e"
        )
        rushing_back_nodes = list(result)
        for r in rushing_back_nodes:
            node = dict(r["e"])
            print(f"  Name: {node.get('name')}")
            print(f"  Type: {node.get('type')}")
            print(f"  Entity ID: {node.get('entity_id')}")
            print(f"  Aliases: {node.get('aliases')}")
            print()

        print("=" * 80)
        print("2. Check if 'Flume' entity exists")
        print("=" * 80)
        result = session.run("MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'flume' RETURN e")
        for r in result:
            node = dict(r["e"])
            print(f"  Name: {node.get('name')}")
            print(f"  Type: {node.get('type')}")
            print(f"  Entity ID: {node.get('entity_id')}")
            print()

        print("=" * 80)
        print("3. Check if 'Vera Blue' entity exists")
        print("=" * 80)
        result = session.run("MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'vera blue' RETURN e")
        for r in result:
            node = dict(r["e"])
            print(f"  Name: {node.get('name')}")
            print(f"  Type: {node.get('type')}")
            print(f"  Entity ID: {node.get('entity_id')}")
            print()

        print("=" * 80)
        print("4. All relations involving 'Rushing Back'")
        print("=" * 80)
        result = session.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
            WHERE toLower(e.name) CONTAINS 'rushing back'
            RETURN e.name AS entity, type(r) AS rel, r.rel_type AS rel_type,
                   other.name AS other_name, other.type AS other_type,
                   startNode(r).name AS src, endNode(r).name AS tgt
            """
        )
        rels = list(result)
        if not rels:
            print("  NO relations found for 'Rushing Back'!")
        for r in rels:
            print(
                f"  {r['src']} --[{r['rel_type']}]--> {r['tgt']}  (other={r['other_name']}, type={r['other_type']})"
            )

        print()
        print("=" * 80)
        print("5. Neighbor expansion (depth=1) for 'Rushing Back' using exact same query as code")
        print("=" * 80)
        # This is the exact query pattern used in graph_repository.py
        for entity_type in ["work", "concept", "event"]:
            result = session.run(
                "MATCH (e:Entity {name: $name, type: $type})-[*1..1]-(neighbor:Entity) "
                "WHERE neighbor <> e RETURN DISTINCT neighbor",
                name="Rushing Back",
                type=entity_type,
            )
            neighbors = list(result)
            if neighbors:
                print(f"  Type={entity_type}: {len(neighbors)} neighbors found")
                for n in neighbors:
                    node = dict(n["neighbor"])
                    print(f"    - {node.get('name')} (type={node.get('type')})")
            else:
                print(f"  Type={entity_type}: no neighbors")

        print()
        print("=" * 80)
        print("6. All relations involving 'Flume'")
        print("=" * 80)
        result = session.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
            WHERE toLower(e.name) CONTAINS 'flume'
            RETURN e.name AS entity, r.rel_type AS rel_type,
                   other.name AS other_name, other.type AS other_type,
                   startNode(r).name AS src, endNode(r).name AS tgt
            """
        )
        rels = list(result)
        if not rels:
            print("  NO relations found for 'Flume'!")
        for r in rels:
            print(f"  {r['src']} --[{r['rel_type']}]--> {r['tgt']}")

        print()
        print("=" * 80)
        print("7. All relations involving 'Vera Blue'")
        print("=" * 80)
        result = session.run(
            """
            MATCH (e:Entity)-[r:RELATES_TO]-(other:Entity)
            WHERE toLower(e.name) CONTAINS 'vera blue'
            RETURN e.name AS entity, r.rel_type AS rel_type,
                   other.name AS other_name, other.type AS other_type,
                   startNode(r).name AS src, endNode(r).name AS tgt
            """
        )
        rels = list(result)
        if not rels:
            print("  NO relations found for 'Vera Blue'!")
        for r in rels:
            print(f"  {r['src']} --[{r['rel_type']}]--> {r['tgt']}")

        print()
        print("=" * 80)
        print("8. Entity type distribution (top types)")
        print("=" * 80)
        result = session.run(
            "MATCH (e:Entity) RETURN e.type AS type, count(*) AS cnt ORDER BY cnt DESC LIMIT 10"
        )
        for r in result:
            print(f"  {r['type']}: {r['cnt']}")

        print()
        print("=" * 80)
        print("9. Total relation count in Neo4j")
        print("=" * 80)
        result = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt")
        record = result.single()
        print(f"  Total RELATES_TO relations: {record['cnt']}")

        print()
        print("=" * 80)
        print("10. Sample relations (first 10)")
        print("=" * 80)
        result = session.run(
            "MATCH (s)-[r:RELATES_TO]->(t) RETURN s.name AS src, r.rel_type AS rel, t.name AS tgt LIMIT 10"
        )
        for r in result:
            print(f"  {r['src']} --[{r['rel']}]--> {r['tgt']}")

        print()
        print("=" * 80)
        print("11. Entities with entity_id like 'work_1' or 'person_1' (old-style IDs)")
        print("=" * 80)
        result = session.run(
            "MATCH (e:Entity) WHERE e.entity_id =~ '(work|person|e)_[0-9]+' RETURN e.name, e.type, e.entity_id LIMIT 20"
        )
        old_style = list(result)
        print(f"  Found {len(old_style)} entities with old-style IDs")
        for r in old_style:
            print(f"    {r['e.entity_id']}: {r['e.name']} ({r['e.type']})")

        print()
        print("=" * 80)
        print("12. Entities with NO relations (orphaned)")
        print("=" * 80)
        result = session.run(
            "MATCH (e:Entity) WHERE NOT (e)-[:RELATES_TO]-() RETURN count(e) AS cnt"
        )
        record = result.single()
        print(f"  Orphaned entities (no relations): {record['cnt']}")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
