"""L2 query result cache with semantic similarity matching via pgvector."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .config import Settings

logger = logging.getLogger(__name__)

_ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key        TEXT PRIMARY KEY,
    query_embedding  vector(1024),
    query_original   TEXT NOT NULL,
    query_rewritten  TEXT,
    answer           TEXT NOT NULL,
    citations        JSONB,
    model_used       TEXT,
    created_at       TIMESTAMPTZ DEFAULT now(),
    accessed_at      TIMESTAMPTZ DEFAULT now(),
    hit_count        INTEGER DEFAULT 0,
    source_doc_ids   TEXT[] DEFAULT '{}'
);
"""

_ENSURE_INDEXES_SQL = [
    """
    CREATE INDEX IF NOT EXISTS idx_query_cache_embedding
    ON query_cache USING ivfflat (query_embedding vector_cosine_ops);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_query_cache_created_at
    ON query_cache(created_at);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_query_cache_source_docs
    ON query_cache USING GIN (source_doc_ids);
    """,
]

_LOOKUP_SQL = """
SELECT
    cache_key,
    query_original,
    query_rewritten,
    answer,
    citations,
    model_used,
    hit_count,
    1 - (query_embedding <=> CAST(:embedding AS vector)) AS similarity
FROM query_cache
WHERE created_at > :min_created_at
ORDER BY query_embedding <=> CAST(:embedding AS vector)
LIMIT 1;
"""

_UPDATE_HIT_SQL = """
UPDATE query_cache
SET hit_count = hit_count + 1,
    accessed_at = now()
WHERE cache_key = :cache_key;
"""

_STORE_SQL = """
INSERT INTO query_cache
    (cache_key, query_embedding, query_original, query_rewritten,
     answer, citations, model_used, source_doc_ids)
VALUES
    (:cache_key, CAST(:embedding AS vector), :query_original, :query_rewritten,
     :answer, CAST(:citations AS jsonb), :model_used, :source_doc_ids)
ON CONFLICT (cache_key) DO UPDATE SET
    answer = EXCLUDED.answer,
    citations = EXCLUDED.citations,
    model_used = EXCLUDED.model_used,
    source_doc_ids = EXCLUDED.source_doc_ids,
    accessed_at = now(),
    created_at = now();
"""

_INVALIDATE_BY_DOC_SQL = """
DELETE FROM query_cache
WHERE source_doc_ids @> CAST(ARRAY[:doc_id] AS TEXT[]);
"""

_CLEANUP_SQL = """
DELETE FROM query_cache
WHERE created_at < :cutoff;
"""


def _make_cache_key(query_original: str, query_rewritten: str | None) -> str:
    canonical = (query_rewritten or query_original).strip().lower()
    return hashlib.sha256(canonical.encode()).hexdigest()


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.9f}" for v in values) + "]"


class QueryCache:
    """Semantic query result cache backed by PostgreSQL + pgvector."""

    def __init__(self, settings: Settings, engine: Engine) -> None:
        self._settings = settings
        self._engine = engine
        self._table_ensured = False

    def ensure_table(self) -> None:
        if self._table_ensured:
            return
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(text(_ENSURE_TABLE_SQL))
            for idx_sql in _ENSURE_INDEXES_SQL:
                try:
                    conn.execute(text(idx_sql))
                except Exception:
                    logger.debug("Index creation skipped (may need more rows for ivfflat)")
        self._table_ensured = True

    def lookup(
        self,
        query_embedding: list[float],
    ) -> dict[str, Any] | None:
        if not self._settings.enable_query_cache:
            return None

        self.ensure_table()
        threshold = self._settings.query_cache_similarity_threshold
        ttl_hours = self._settings.query_cache_ttl_hours
        min_created = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        vec = _vector_literal(query_embedding)

        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    text(_LOOKUP_SQL),
                    {"embedding": vec, "min_created_at": min_created},
                )
                .mappings()
                .first()
            )

        if row is None:
            logger.info("query_cache MISS (no rows)")
            return None

        similarity = float(row["similarity"])
        if similarity < threshold:
            logger.info(
                "query_cache MISS (similarity=%.4f < threshold=%.4f)",
                similarity,
                threshold,
            )
            return None

        cache_key = row["cache_key"]
        with self._engine.begin() as conn:
            conn.execute(text(_UPDATE_HIT_SQL), {"cache_key": cache_key})

        citations = row["citations"]
        if isinstance(citations, str):
            citations = json.loads(citations)

        logger.info(
            "query_cache HIT (similarity=%.4f, hit_count=%d, key=%s)",
            similarity,
            int(row["hit_count"]) + 1,
            cache_key[:12],
        )
        return {
            "answer": row["answer"],
            "citations": citations or [],
            "model_used": row["model_used"],
            "cache_key": cache_key,
            "similarity": similarity,
        }

    def store(
        self,
        query_original: str,
        query_rewritten: str | None,
        query_embedding: list[float],
        answer: str,
        citations: list[dict[str, Any]],
        model_used: str,
        source_doc_ids: list[str] | None = None,
    ) -> str:
        if not self._settings.enable_query_cache:
            return ""

        self.ensure_table()
        cache_key = _make_cache_key(query_original, query_rewritten)
        vec = _vector_literal(query_embedding)

        with self._engine.begin() as conn:
            conn.execute(
                text(_STORE_SQL),
                {
                    "cache_key": cache_key,
                    "embedding": vec,
                    "query_original": query_original,
                    "query_rewritten": query_rewritten or "",
                    "answer": answer,
                    "citations": json.dumps(citations),
                    "model_used": model_used or "",
                    "source_doc_ids": source_doc_ids or [],
                },
            )

        logger.info("query_cache STORE (key=%s)", cache_key[:12])
        return cache_key

    def invalidate_by_doc(self, doc_id: str) -> int:
        self.ensure_table()
        with self._engine.begin() as conn:
            result = conn.execute(text(_INVALIDATE_BY_DOC_SQL), {"doc_id": doc_id})
            deleted = result.rowcount
        logger.info("query_cache INVALIDATE doc=%s deleted=%d", doc_id, deleted)
        return deleted

    def cleanup_expired(self) -> int:
        self.ensure_table()
        ttl_hours = self._settings.query_cache_ttl_hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        with self._engine.begin() as conn:
            result = conn.execute(text(_CLEANUP_SQL), {"cutoff": cutoff})
            deleted = result.rowcount
        logger.info("query_cache CLEANUP deleted=%d (ttl=%dh)", deleted, ttl_hours)
        return deleted
