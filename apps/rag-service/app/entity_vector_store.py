"""Write-path repository for entity/relation vector storage in PostgreSQL + pgvector.

Handles INSERT/UPDATE operations and similarity search for extracted entities and relations.
Uses SQLAlchemy Core (not ORM) and mirrors the lazy engine pattern from ingestion_repository.py.

Tables: kb_entities, kb_relations (see docs/entity-vector-store-ddl.sql for schema).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import Settings
from .secrets import resolve_db_password

logger = logging.getLogger(__name__)


def _to_vector_literal(values: list[float]) -> str:
    """Convert a Python float list to PostgreSQL pgvector literal format."""

    return "[" + ",".join(f"{v:.9f}" for v in values) + "]"


# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_UPSERT_ENTITY_SQL = text(
    """
    INSERT INTO kb_entities (
        entity_id, name, type, canonical_key, description,
        aliases, confidence, source_chunk_ids, embedding, metadata
    )
    VALUES (
        :entity_id, :name, :type, :canonical_key, :description,
        :aliases, :confidence, :source_chunk_ids,
        CAST(:embedding AS vector), CAST(:metadata AS jsonb)
    )
    ON CONFLICT (entity_id) DO UPDATE
        SET name = EXCLUDED.name,
            type = EXCLUDED.type,
            canonical_key = EXCLUDED.canonical_key,
            description = CASE
                WHEN length(EXCLUDED.description) > length(kb_entities.description)
                THEN EXCLUDED.description
                ELSE kb_entities.description
            END,
            aliases = (
                SELECT array_agg(DISTINCT val)
                FROM unnest(kb_entities.aliases || EXCLUDED.aliases) AS val
            ),
            confidence = GREATEST(kb_entities.confidence, EXCLUDED.confidence),
            source_chunk_ids = (
                SELECT array_agg(DISTINCT val)
                FROM unnest(kb_entities.source_chunk_ids || EXCLUDED.source_chunk_ids) AS val
            ),
            embedding = EXCLUDED.embedding,
            metadata = kb_entities.metadata || EXCLUDED.metadata
    RETURNING entity_id
    """
)

_UPSERT_RELATION_SQL = text(
    """
    INSERT INTO kb_relations (
        relation_id, source_entity_id, target_entity_id, type,
        evidence, confidence, weight, source_chunk_ids, embedding, metadata
    )
    VALUES (
        :relation_id, :source_entity_id, :target_entity_id, :type,
        :evidence, :confidence, :weight, :source_chunk_ids,
        CAST(:embedding AS vector), CAST(:metadata AS jsonb)
    )
    ON CONFLICT (relation_id) DO UPDATE
        SET evidence = CASE
                WHEN kb_relations.evidence = '' THEN EXCLUDED.evidence
                WHEN EXCLUDED.evidence = '' THEN kb_relations.evidence
                ELSE kb_relations.evidence || ' | ' || EXCLUDED.evidence
            END,
            confidence = GREATEST(kb_relations.confidence, EXCLUDED.confidence),
            weight = kb_relations.weight + EXCLUDED.weight,
            source_chunk_ids = (
                SELECT array_agg(DISTINCT val)
                FROM unnest(kb_relations.source_chunk_ids || EXCLUDED.source_chunk_ids) AS val
            ),
            embedding = EXCLUDED.embedding,
            metadata = kb_relations.metadata || EXCLUDED.metadata
    RETURNING relation_id
    """
)

_SEARCH_ENTITIES_SQL = text(
    """
    SELECT
        entity_id, name, type, canonical_key, description,
        aliases, confidence, source_chunk_ids,
        embedding <=> CAST(:query_embedding AS vector) AS distance
    FROM kb_entities
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :top_k
    """
)

_SEARCH_RELATIONS_SQL = text(
    """
    SELECT
        relation_id, source_entity_id, target_entity_id, type,
        evidence, confidence, weight, source_chunk_ids,
        embedding <=> CAST(:query_embedding AS vector) AS distance
    FROM kb_relations
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:query_embedding AS vector)
    LIMIT :top_k
    """
)

_GET_ENTITY_SQL = text(
    """
    SELECT
        entity_id, name, type, canonical_key, description,
        aliases, confidence, source_chunk_ids
    FROM kb_entities
    WHERE entity_id = :entity_id
    """
)

_GET_ENTITIES_BY_NAME_SQL = text(
    """
    SELECT
        entity_id, name, type, canonical_key, description,
        aliases, confidence, source_chunk_ids
    FROM kb_entities
    WHERE lower(name) = lower(:name)
    """
)

_DELETE_ENTITIES_BY_SOURCE_SQL = text(
    """
    DELETE FROM kb_entities
    WHERE source_chunk_ids && :chunk_ids
    RETURNING entity_id
    """
)


class EntityVectorStore:
    """Repository for entity/relation vector storage and similarity search.

    Uses PostgreSQL + pgvector for embedding storage and cosine distance search.
    Follows the same lazy engine pattern as IngestionRepository.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: Engine | None = None

    def _get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine

        db_password = resolve_db_password(self._settings)
        if not db_password:
            raise ValueError(
                "DB password is not configured. Set RAG_DB_PASSWORD or "
                "RAG_DB_PASSWORD_SECRET_ARN."
            )

        self._engine = create_engine(
            self._settings.build_db_dsn(db_password),
            pool_pre_ping=True,
        )
        return self._engine

    def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        embedding: list[float],
        *,
        canonical_key: str | None = None,
        aliases: list[str] | None = None,
        confidence: float = 0.0,
        source_chunk_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Insert or update an entity with its embedding vector.

        On conflict (entity_id), merges description (keeps longer), aliases (union),
        confidence (keeps higher), and source_chunk_ids (union).

        Returns the entity_id on success, None on failure.
        """

        import json

        params = {
            "entity_id": entity_id,
            "name": name,
            "type": entity_type,
            "canonical_key": canonical_key or "",
            "description": description,
            "aliases": aliases or [],
            "confidence": confidence,
            "source_chunk_ids": source_chunk_ids or [],
            "embedding": _to_vector_literal(embedding),
            "metadata": json.dumps(metadata or {}),
        }

        engine = self._get_engine()
        try:
            with engine.begin() as conn:
                row = conn.execute(_UPSERT_ENTITY_SQL, params).fetchone()
            return str(row[0]) if row else None
        except Exception:
            logger.exception("Failed to upsert entity %s", entity_id)
            return None

    def upsert_relation(
        self,
        relation_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        evidence: str,
        embedding: list[float],
        *,
        confidence: float = 0.0,
        weight: float = 1.0,
        source_chunk_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Insert or update a relation with its embedding vector.

        On conflict (relation_id), merges evidence (concatenated with ' | '),
        confidence (keeps higher), weight (accumulated), and source_chunk_ids (union).

        Returns the relation_id on success, None on failure.
        """

        import json

        params = {
            "relation_id": relation_id,
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "type": relation_type,
            "evidence": evidence,
            "confidence": confidence,
            "weight": weight,
            "source_chunk_ids": source_chunk_ids or [],
            "embedding": _to_vector_literal(embedding),
            "metadata": json.dumps(metadata or {}),
        }

        engine = self._get_engine()
        try:
            with engine.begin() as conn:
                row = conn.execute(_UPSERT_RELATION_SQL, params).fetchone()
            return str(row[0]) if row else None
        except Exception:
            logger.exception("Failed to upsert relation %s", relation_id)
            return None

    def search_entities(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find entities by cosine similarity to query embedding.

        Returns list of dicts with entity fields + 'distance' (lower = more similar).
        """

        params = {
            "query_embedding": _to_vector_literal(query_embedding),
            "top_k": top_k,
        }

        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(_SEARCH_ENTITIES_SQL, params).fetchall()

        return [
            {
                "entity_id": row[0],
                "name": row[1],
                "type": row[2],
                "canonical_key": row[3],
                "description": row[4],
                "aliases": list(row[5]) if row[5] else [],
                "confidence": float(row[6]),
                "source_chunk_ids": list(row[7]) if row[7] else [],
                "distance": float(row[8]),
            }
            for row in rows
        ]

    def search_relations(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Find relations by cosine similarity to query embedding.

        Returns list of dicts with relation fields + 'distance' (lower = more similar).
        """

        params = {
            "query_embedding": _to_vector_literal(query_embedding),
            "top_k": top_k,
        }

        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(_SEARCH_RELATIONS_SQL, params).fetchall()

        return [
            {
                "relation_id": row[0],
                "source_entity_id": row[1],
                "target_entity_id": row[2],
                "type": row[3],
                "evidence": row[4],
                "confidence": float(row[5]),
                "weight": float(row[6]),
                "source_chunk_ids": list(row[7]) if row[7] else [],
                "distance": float(row[8]),
            }
            for row in rows
        ]

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Fetch a single entity by entity_id. Returns None if not found."""

        engine = self._get_engine()
        with engine.connect() as conn:
            row = conn.execute(_GET_ENTITY_SQL, {"entity_id": entity_id}).fetchone()

        if row is None:
            return None

        return {
            "entity_id": row[0],
            "name": row[1],
            "type": row[2],
            "canonical_key": row[3],
            "description": row[4],
            "aliases": list(row[5]) if row[5] else [],
            "confidence": float(row[6]),
            "source_chunk_ids": list(row[7]) if row[7] else [],
        }

    def get_entities_by_name(self, name: str) -> list[dict[str, Any]]:
        """Fetch entities by name (case-insensitive). Returns empty list if not found."""

        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(_GET_ENTITIES_BY_NAME_SQL, {"name": name}).fetchall()

        return [
            {
                "entity_id": row[0],
                "name": row[1],
                "type": row[2],
                "canonical_key": row[3],
                "description": row[4],
                "aliases": list(row[5]) if row[5] else [],
                "confidence": float(row[6]),
                "source_chunk_ids": list(row[7]) if row[7] else [],
            }
            for row in rows
        ]

    def delete_entities_by_source_chunks(self, chunk_ids: list[str]) -> list[str]:
        """Delete entities whose source_chunk_ids overlap with the given chunk_ids.

        Relations referencing deleted entities are cascade-deleted via FK constraint.
        Returns list of deleted entity_ids.
        """

        if not chunk_ids:
            return []

        engine = self._get_engine()
        with engine.begin() as conn:
            rows = conn.execute(_DELETE_ENTITIES_BY_SOURCE_SQL, {"chunk_ids": chunk_ids}).fetchall()

        return [str(row[0]) for row in rows]

    def batch_upsert_entities(
        self,
        entities: list[dict[str, Any]],
    ) -> int:
        """Batch upsert multiple entities. Returns count of successfully upserted entities.

        Each dict must have: entity_id, name, type, description, embedding.
        Optional: canonical_key, aliases, confidence, source_chunk_ids, metadata.
        """

        if not entities:
            return 0

        count = 0
        for entity in entities:
            result = self.upsert_entity(
                entity_id=entity["entity_id"],
                name=entity["name"],
                entity_type=entity["type"],
                description=entity.get("description", ""),
                embedding=entity["embedding"],
                canonical_key=entity.get("canonical_key"),
                aliases=entity.get("aliases"),
                confidence=entity.get("confidence", 0.0),
                source_chunk_ids=entity.get("source_chunk_ids"),
                metadata=entity.get("metadata"),
            )
            if result is not None:
                count += 1
        return count

    def batch_upsert_relations(
        self,
        relations: list[dict[str, Any]],
    ) -> int:
        """Batch upsert multiple relations. Returns count of successfully upserted relations.

        Each dict must have: relation_id, source_entity_id, target_entity_id, type,
        evidence, embedding.
        Optional: confidence, weight, source_chunk_ids, metadata.
        """

        if not relations:
            return 0

        count = 0
        for relation in relations:
            result = self.upsert_relation(
                relation_id=relation["relation_id"],
                source_entity_id=relation["source_entity_id"],
                target_entity_id=relation["target_entity_id"],
                relation_type=relation["type"],
                evidence=relation.get("evidence", ""),
                embedding=relation["embedding"],
                confidence=relation.get("confidence", 0.0),
                weight=relation.get("weight", 1.0),
                source_chunk_ids=relation.get("source_chunk_ids"),
                metadata=relation.get("metadata"),
            )
            if result is not None:
                count += 1
        return count
