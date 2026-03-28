"""Neo4j graph repository for entity and relation storage.

Provides CRUD and traversal operations on the knowledge graph built from
extracted entities and relations (Phase 2.3 of LightRAG migration).

The repository uses the synchronous neo4j Python driver with managed
transactions (auto-retry on TransientError). Connection is lazily initialized
on first use and must be explicitly closed via ``close()``.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction

from .entity_extraction_models import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher query constants
# ---------------------------------------------------------------------------

# Upsert a single entity node.
# MERGE on (name, type) to ensure uniqueness; ON CREATE sets initial fields,
# ON MATCH accumulates aliases and source_chunk_ids.
_UPSERT_ENTITY_CYPHER = """
MERGE (e:Entity {name: $name, type: $type})
ON CREATE SET
    e.entity_id        = $entity_id,
    e.canonical_key    = $canonical_key,
    e.description      = $description,
    e.aliases          = $aliases,
    e.confidence       = $confidence,
    e.source_chunk_ids = $source_chunk_ids
ON MATCH SET
    e.description      = CASE WHEN size(e.description) < size($description)
                              THEN $description ELSE e.description END,
    e.aliases          = apoc.coll.toSet(e.aliases + $aliases),
    e.confidence       = CASE WHEN $confidence > e.confidence
                              THEN $confidence ELSE e.confidence END,
    e.source_chunk_ids = apoc.coll.toSet(e.source_chunk_ids + $source_chunk_ids)
RETURN e.entity_id AS entity_id
"""

# Upsert a single relation.
# MERGE on source name+type and target name+type and relation type.
_UPSERT_RELATION_CYPHER = """
MATCH (src:Entity {name: $source_name, type: $source_type})
MATCH (tgt:Entity {name: $target_name, type: $target_type})
MERGE (src)-[r:RELATES_TO {rel_type: $rel_type}]->(tgt)
ON CREATE SET
    r.evidence         = $evidence,
    r.confidence       = $confidence,
    r.weight           = $weight,
    r.source_chunk_ids = $source_chunk_ids
ON MATCH SET
    r.evidence         = r.evidence + ' | ' + $evidence,
    r.confidence       = CASE WHEN $confidence > r.confidence
                              THEN $confidence ELSE r.confidence END,
    r.weight           = r.weight + $weight,
    r.source_chunk_ids = apoc.coll.toSet(r.source_chunk_ids + $source_chunk_ids)
RETURN type(r) AS rel
"""

# Batch upsert entities via UNWIND.
_BATCH_UPSERT_ENTITIES_CYPHER = """
UNWIND $batch AS row
MERGE (e:Entity {name: row.name, type: row.type})
ON CREATE SET
    e.entity_id        = row.entity_id,
    e.canonical_key    = row.canonical_key,
    e.description      = row.description,
    e.aliases          = row.aliases,
    e.confidence       = row.confidence,
    e.source_chunk_ids = row.source_chunk_ids
ON MATCH SET
    e.description      = CASE WHEN size(e.description) < size(row.description)
                              THEN row.description ELSE e.description END,
    e.aliases          = apoc.coll.toSet(e.aliases + row.aliases),
    e.confidence       = CASE WHEN row.confidence > e.confidence
                              THEN row.confidence ELSE e.confidence END,
    e.source_chunk_ids = apoc.coll.toSet(e.source_chunk_ids + row.source_chunk_ids)
"""

# Batch upsert relations via UNWIND.
_BATCH_UPSERT_RELATIONS_CYPHER = """
UNWIND $batch AS row
MATCH (src:Entity {name: row.source_name, type: row.source_type})
MATCH (tgt:Entity {name: row.target_name, type: row.target_type})
MERGE (src)-[r:RELATES_TO {rel_type: row.rel_type}]->(tgt)
ON CREATE SET
    r.evidence         = row.evidence,
    r.confidence       = row.confidence,
    r.weight           = row.weight,
    r.source_chunk_ids = row.source_chunk_ids
ON MATCH SET
    r.evidence         = r.evidence + ' | ' + row.evidence,
    r.confidence       = CASE WHEN row.confidence > r.confidence
                              THEN row.confidence ELSE r.confidence END,
    r.weight           = r.weight + row.weight,
    r.source_chunk_ids = apoc.coll.toSet(r.source_chunk_ids + row.source_chunk_ids)
"""

_GET_ENTITY_CYPHER = """
MATCH (e:Entity {name: $name, type: $type})
RETURN e
"""

# Retrieve entity + neighbors up to N hops.
_GET_NEIGHBORS_CYPHER = """
MATCH (e:Entity {name: $name, type: $type})
CALL apoc.path.subgraphNodes(e, {maxLevel: $depth}) YIELD node
WITH node WHERE node <> e
RETURN node
"""

# Retrieve all relations between a set of entity names.
_GET_RELATIONS_FOR_ENTITIES_CYPHER = """
MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
WHERE src.name IN $entity_names AND tgt.name IN $entity_names
RETURN src.name AS source_name, src.type AS source_type,
       tgt.name AS target_name, tgt.type AS target_type,
       r.rel_type AS rel_type, r.evidence AS evidence,
       r.confidence AS confidence, r.weight AS weight,
       r.source_chunk_ids AS source_chunk_ids
"""

# Full-text search on entity name (requires a full-text index named 'entity_name_fulltext').
_SEARCH_FULLTEXT_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query) YIELD node, score
RETURN node, score
ORDER BY score DESC
LIMIT $limit
"""

# Index creation statements.
_CREATE_BTREE_INDEX = "CREATE INDEX entity_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)"
_CREATE_COMPOSITE_INDEX = (
    "CREATE INDEX entity_name_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.name, e.type)"
)
_CREATE_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS " "FOR (e:Entity) ON EACH [e.name]"
)


# ---------------------------------------------------------------------------
# Neo4j Community Edition compatibility
# ---------------------------------------------------------------------------
# Neo4j Community does not include APOC by default. The Cypher queries above
# use ``apoc.coll.toSet`` and ``apoc.path.subgraphNodes``. When APOC is not
# available we fall back to plain Cypher equivalents (at the cost of potential
# duplicates in list properties that the caller can deduplicate in Python).

_UPSERT_ENTITY_CYPHER_NO_APOC = """
MERGE (e:Entity {name: $name, type: $type})
ON CREATE SET
    e.entity_id        = $entity_id,
    e.canonical_key    = $canonical_key,
    e.description      = $description,
    e.aliases          = $aliases,
    e.confidence       = $confidence,
    e.source_chunk_ids = $source_chunk_ids
ON MATCH SET
    e.description      = CASE WHEN size(e.description) < size($description)
                              THEN $description ELSE e.description END,
    e.aliases          = e.aliases + [x IN $aliases WHERE NOT x IN e.aliases],
    e.confidence       = CASE WHEN $confidence > e.confidence
                              THEN $confidence ELSE e.confidence END,
    e.source_chunk_ids = e.source_chunk_ids + [x IN $source_chunk_ids WHERE NOT x IN e.source_chunk_ids]
RETURN e.entity_id AS entity_id
"""

_UPSERT_RELATION_CYPHER_NO_APOC = """
MATCH (src:Entity {name: $source_name, type: $source_type})
MATCH (tgt:Entity {name: $target_name, type: $target_type})
MERGE (src)-[r:RELATES_TO {rel_type: $rel_type}]->(tgt)
ON CREATE SET
    r.evidence         = $evidence,
    r.confidence       = $confidence,
    r.weight           = $weight,
    r.source_chunk_ids = $source_chunk_ids
ON MATCH SET
    r.evidence         = r.evidence + ' | ' + $evidence,
    r.confidence       = CASE WHEN $confidence > r.confidence
                              THEN $confidence ELSE r.confidence END,
    r.weight           = r.weight + $weight,
    r.source_chunk_ids = r.source_chunk_ids + [x IN $source_chunk_ids WHERE NOT x IN r.source_chunk_ids]
RETURN type(r) AS rel
"""

_BATCH_UPSERT_ENTITIES_CYPHER_NO_APOC = """
UNWIND $batch AS row
MERGE (e:Entity {name: row.name, type: row.type})
ON CREATE SET
    e.entity_id        = row.entity_id,
    e.canonical_key    = row.canonical_key,
    e.description      = row.description,
    e.aliases          = row.aliases,
    e.confidence       = row.confidence,
    e.source_chunk_ids = row.source_chunk_ids
ON MATCH SET
    e.description      = CASE WHEN size(e.description) < size(row.description)
                              THEN row.description ELSE e.description END,
    e.aliases          = e.aliases + [x IN row.aliases WHERE NOT x IN e.aliases],
    e.confidence       = CASE WHEN row.confidence > e.confidence
                              THEN row.confidence ELSE e.confidence END,
    e.source_chunk_ids = e.source_chunk_ids + [x IN row.source_chunk_ids WHERE NOT x IN e.source_chunk_ids]
"""

_BATCH_UPSERT_RELATIONS_CYPHER_NO_APOC = """
UNWIND $batch AS row
MATCH (src:Entity {name: row.source_name, type: row.source_type})
MATCH (tgt:Entity {name: row.target_name, type: row.target_type})
MERGE (src)-[r:RELATES_TO {rel_type: row.rel_type}]->(tgt)
ON CREATE SET
    r.evidence         = row.evidence,
    r.confidence       = row.confidence,
    r.weight           = row.weight,
    r.source_chunk_ids = row.source_chunk_ids
ON MATCH SET
    r.evidence         = r.evidence + ' | ' + row.evidence,
    r.confidence       = CASE WHEN row.confidence > r.confidence
                              THEN row.confidence ELSE r.confidence END,
    r.weight           = r.weight + row.weight,
    r.source_chunk_ids = r.source_chunk_ids + [x IN row.source_chunk_ids WHERE NOT x IN r.source_chunk_ids]
"""

# Neighbor traversal without APOC — uses variable-length path pattern.
_GET_NEIGHBORS_CYPHER_NO_APOC = """
MATCH (e:Entity {name: $name, type: $type})-[*1..$depth]-(neighbor:Entity)
WHERE neighbor <> e
RETURN DISTINCT neighbor
"""


def _entity_to_params(entity: ExtractedEntity) -> dict[str, Any]:
    """Convert an ExtractedEntity to Neo4j query parameters."""
    return {
        "entity_id": entity.entity_id,
        "name": entity.name,
        "type": entity.type.value,
        "canonical_key": entity.canonical_key or "",
        "description": entity.description,
        "aliases": entity.aliases,
        "confidence": entity.confidence,
        "source_chunk_ids": entity.source_chunk_ids,
    }


def _relation_to_params(
    relation: ExtractedRelation,
    entity_map: dict[str, ExtractedEntity],
) -> dict[str, Any] | None:
    """Convert an ExtractedRelation to Neo4j query parameters.

    Returns None if source or target entity is not found in entity_map.
    """
    src = entity_map.get(relation.source_entity_id)
    tgt = entity_map.get(relation.target_entity_id)
    if src is None or tgt is None:
        logger.warning(
            "Skipping relation %s->%s: missing entity in map",
            relation.source_entity_id,
            relation.target_entity_id,
        )
        return None
    return {
        "source_name": src.name,
        "source_type": src.type.value,
        "target_name": tgt.name,
        "target_type": tgt.type.value,
        "rel_type": relation.type.value,
        "evidence": relation.evidence,
        "confidence": relation.confidence,
        "weight": relation.weight,
        "source_chunk_ids": relation.source_chunk_ids,
    }


def _node_to_entity(node: Any) -> ExtractedEntity:
    """Convert a Neo4j node record to an ExtractedEntity."""
    props = dict(node) if not isinstance(node, dict) else node
    return ExtractedEntity(
        entity_id=props.get("entity_id", ""),
        type=EntityType(props.get("type", "Work")),
        name=props.get("name", ""),
        canonical_key=props.get("canonical_key") or None,
        description=props.get("description", ""),
        aliases=list(props.get("aliases", [])),
        confidence=float(props.get("confidence", 0.0)),
        source_chunk_ids=list(props.get("source_chunk_ids", [])),
    )


class Neo4jRepository:
    """Synchronous Neo4j repository for knowledge graph CRUD and traversal.

    Connection is lazily initialized on first operation. The caller must invoke
    ``close()`` when the repository is no longer needed (e.g., at application
    shutdown).

    Args:
        uri: Neo4j Bolt URI (e.g., ``bolt://localhost:7687``).
        username: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name (default ``neo4j``).
        use_apoc: Whether the Neo4j instance has APOC installed. When False,
            plain Cypher fallbacks are used for list dedup and neighbor
            traversal.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        *,
        use_apoc: bool = False,
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._use_apoc = use_apoc
        self._driver = None

    # -- connection management --------------------------------------------------

    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._username, self._password),
            )
            logger.info("Neo4j driver created for %s", self._uri)
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    # -- write operations -------------------------------------------------------

    def upsert_entity(self, entity: ExtractedEntity) -> str | None:
        """Upsert a single entity node. Returns the entity_id on success."""
        params = _entity_to_params(entity)
        cypher = _UPSERT_ENTITY_CYPHER if self._use_apoc else _UPSERT_ENTITY_CYPHER_NO_APOC

        def _work(tx: ManagedTransaction) -> str | None:
            result = tx.run(cypher, **params)
            record = result.single()
            return record["entity_id"] if record else None

        with self._get_driver().session(database=self._database) as session:
            return session.execute_write(_work)

    def upsert_relation(
        self,
        relation: ExtractedRelation,
        entity_map: dict[str, ExtractedEntity],
    ) -> bool:
        """Upsert a single relation. Returns True if the relation was written."""
        params = _relation_to_params(relation, entity_map)
        if params is None:
            return False
        cypher = _UPSERT_RELATION_CYPHER if self._use_apoc else _UPSERT_RELATION_CYPHER_NO_APOC

        def _work(tx: ManagedTransaction) -> bool:
            result = tx.run(cypher, **params)
            return result.single() is not None

        with self._get_driver().session(database=self._database) as session:
            return session.execute_write(_work)

    def upsert_batch(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> dict[str, int]:
        """Batch upsert entities and relations in two UNWIND transactions.

        Returns a dict with ``entities_written`` and ``relations_written`` counts.
        """
        entity_map = {e.entity_id: e for e in entities}

        entity_batch = [_entity_to_params(e) for e in entities]
        relation_batch = [
            p for r in relations if (p := _relation_to_params(r, entity_map)) is not None
        ]

        entity_cypher = (
            _BATCH_UPSERT_ENTITIES_CYPHER
            if self._use_apoc
            else _BATCH_UPSERT_ENTITIES_CYPHER_NO_APOC
        )
        relation_cypher = (
            _BATCH_UPSERT_RELATIONS_CYPHER
            if self._use_apoc
            else _BATCH_UPSERT_RELATIONS_CYPHER_NO_APOC
        )

        written = {"entities_written": 0, "relations_written": 0}

        def _write_entities(tx: ManagedTransaction) -> int:
            result = tx.run(entity_cypher, batch=entity_batch)
            summary = result.consume()
            return summary.counters.nodes_created + summary.counters.properties_set

        def _write_relations(tx: ManagedTransaction) -> int:
            result = tx.run(relation_cypher, batch=relation_batch)
            summary = result.consume()
            return summary.counters.relationships_created + summary.counters.properties_set

        with self._get_driver().session(database=self._database) as session:
            if entity_batch:
                _write_entities_result = session.execute_write(_write_entities)
                written["entities_written"] = len(entity_batch)
                logger.info(
                    "Batch upserted %d entities (neo4j counters: %d)",
                    len(entity_batch),
                    _write_entities_result,
                )
            if relation_batch:
                _write_relations_result = session.execute_write(_write_relations)
                written["relations_written"] = len(relation_batch)
                logger.info(
                    "Batch upserted %d relations (neo4j counters: %d)",
                    len(relation_batch),
                    _write_relations_result,
                )

        return written

    # -- read operations --------------------------------------------------------

    def get_entity(self, name: str, entity_type: EntityType) -> ExtractedEntity | None:
        """Fetch a single entity by name and type. Returns None if not found."""

        def _work(tx: ManagedTransaction) -> ExtractedEntity | None:
            result = tx.run(_GET_ENTITY_CYPHER, name=name, type=entity_type.value)
            record = result.single()
            if record is None:
                return None
            return _node_to_entity(record["e"])

        with self._get_driver().session(database=self._database) as session:
            return session.execute_read(_work)

    def get_entity_neighbors(
        self,
        name: str,
        entity_type: EntityType,
        depth: int = 1,
    ) -> list[ExtractedEntity]:
        """Retrieve neighboring entities up to ``depth`` hops away."""
        cypher = _GET_NEIGHBORS_CYPHER if self._use_apoc else _GET_NEIGHBORS_CYPHER_NO_APOC

        def _work(tx: ManagedTransaction) -> list[ExtractedEntity]:
            result = tx.run(
                cypher,
                name=name,
                type=entity_type.value,
                depth=depth,
            )
            return [
                _node_to_entity(record["node" if self._use_apoc else "neighbor"])
                for record in result
            ]

        with self._get_driver().session(database=self._database) as session:
            return session.execute_read(_work)

    def get_relations_for_entities(
        self,
        entity_names: list[str],
    ) -> list[dict[str, Any]]:
        """Get all relations between a set of entity names.

        Returns a list of dicts with keys: source_name, source_type,
        target_name, target_type, rel_type, evidence, confidence, weight,
        source_chunk_ids.
        """

        def _work(tx: ManagedTransaction) -> list[dict[str, Any]]:
            result = tx.run(
                _GET_RELATIONS_FOR_ENTITIES_CYPHER,
                entity_names=entity_names,
            )
            return [dict(record) for record in result]

        with self._get_driver().session(database=self._database) as session:
            return session.execute_read(_work)

    def search_entities_fulltext(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[ExtractedEntity, float]]:
        """Search entities by name using the full-text index.

        Returns a list of (entity, score) tuples sorted by relevance.
        """

        def _work(tx: ManagedTransaction) -> list[tuple[ExtractedEntity, float]]:
            result = tx.run(
                _SEARCH_FULLTEXT_CYPHER,
                query=query,
                limit=limit,
            )
            return [(_node_to_entity(record["node"]), float(record["score"])) for record in result]

        with self._get_driver().session(database=self._database) as session:
            return session.execute_read(_work)

    # -- schema / admin ---------------------------------------------------------

    def ensure_indexes(self) -> None:
        """Create indexes if they do not already exist."""
        with self._get_driver().session(database=self._database) as session:
            for stmt in (
                _CREATE_BTREE_INDEX,
                _CREATE_COMPOSITE_INDEX,
                _CREATE_FULLTEXT_INDEX,
            ):
                session.run(stmt)
        logger.info("Neo4j indexes ensured")

    def health_check(self) -> bool:
        """Verify connectivity to Neo4j. Returns True if healthy."""
        try:
            self._get_driver().verify_connectivity()
            return True
        except Exception:
            logger.exception("Neo4j health check failed")
            return False
