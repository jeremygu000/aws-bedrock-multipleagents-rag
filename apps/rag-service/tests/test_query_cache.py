from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.query_cache import QueryCache, _make_cache_key, _vector_literal


class FakeConnection:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []
        self.executed: list[tuple[str, dict]] = []

    def execute(self, statement: Any, params: Any = None) -> "FakeResult":
        sql = str(statement) if not isinstance(statement, str) else statement
        self.executed.append((sql, params or {}))
        return FakeResult(self._rows)

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.rowcount = len(rows)

    def mappings(self) -> "FakeResult":
        return self

    def all(self) -> list[dict[str, Any]]:
        return self._rows

    def first(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class FakeEngine:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []
        self.connections: list[FakeConnection] = []

    def connect(self) -> FakeConnection:
        conn = FakeConnection(self._rows)
        self.connections.append(conn)
        return conn

    def begin(self) -> FakeConnection:
        conn = FakeConnection(self._rows)
        self.connections.append(conn)
        return conn


def _settings(**overrides: str) -> Settings:
    defaults = {
        "RAG_ENABLE_QUERY_CACHE": "true",
        "RAG_QUERY_CACHE_TTL_HOURS": "24",
        "RAG_QUERY_CACHE_SIMILARITY_THRESHOLD": "0.95",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _embedding(dim: int = 3) -> list[float]:
    return [0.1 * i for i in range(dim)]


class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        k1 = _make_cache_key("hello world", None)
        k2 = _make_cache_key("hello world", None)
        assert k1 == k2

    def test_uses_rewritten_when_provided(self) -> None:
        k1 = _make_cache_key("original", "rewritten")
        k2 = _make_cache_key("different", "rewritten")
        assert k1 == k2

    def test_case_insensitive(self) -> None:
        k1 = _make_cache_key("Hello World", None)
        k2 = _make_cache_key("hello world", None)
        assert k1 == k2

    def test_strips_whitespace(self) -> None:
        k1 = _make_cache_key("  hello  ", None)
        k2 = _make_cache_key("hello", None)
        assert k1 == k2


class TestVectorLiteral:
    def test_format(self) -> None:
        result = _vector_literal([1.0, 2.5, 3.0])
        assert result.startswith("[")
        assert result.endswith("]")
        assert "1.000000000" in result
        assert "2.500000000" in result


class TestQueryCacheLookup:
    def test_returns_none_when_disabled(self) -> None:
        settings = _settings(RAG_ENABLE_QUERY_CACHE="false")
        cache = QueryCache(settings=settings, engine=FakeEngine())
        assert cache.lookup(_embedding()) is None

    def test_returns_none_on_no_rows(self) -> None:
        engine = FakeEngine(rows=[])
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        result = cache.lookup(_embedding())
        assert result is None

    def test_returns_none_below_threshold(self) -> None:
        row = {
            "cache_key": "abc",
            "query_original": "q",
            "query_rewritten": "q",
            "answer": "a",
            "citations": "[]",
            "model_used": "nova-lite",
            "hit_count": 0,
            "similarity": 0.90,
        }
        engine = FakeEngine(rows=[row])
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        result = cache.lookup(_embedding())
        assert result is None

    def test_returns_result_above_threshold(self) -> None:
        row = {
            "cache_key": "abc123",
            "query_original": "q",
            "query_rewritten": "q",
            "answer": "cached answer",
            "citations": json.dumps([{"sourceId": "c1"}]),
            "model_used": "nova-lite",
            "hit_count": 3,
            "similarity": 0.98,
        }
        engine = FakeEngine(rows=[row])
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        result = cache.lookup(_embedding())
        assert result is not None
        assert result["answer"] == "cached answer"
        assert result["citations"] == [{"sourceId": "c1"}]
        assert result["similarity"] == 0.98

    def test_returns_result_with_dict_citations(self) -> None:
        row = {
            "cache_key": "abc",
            "query_original": "q",
            "query_rewritten": "q",
            "answer": "a",
            "citations": [{"sourceId": "c1"}],
            "model_used": "m",
            "hit_count": 0,
            "similarity": 0.99,
        }
        engine = FakeEngine(rows=[row])
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        result = cache.lookup(_embedding())
        assert result is not None
        assert result["citations"] == [{"sourceId": "c1"}]

    def test_graceful_on_lookup_exception(self) -> None:
        engine = MagicMock()
        engine.connect.side_effect = RuntimeError("db down")
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        with pytest.raises(RuntimeError):
            cache.lookup(_embedding())


class TestQueryCacheStore:
    def test_returns_empty_when_disabled(self) -> None:
        settings = _settings(RAG_ENABLE_QUERY_CACHE="false")
        cache = QueryCache(settings=settings, engine=FakeEngine())
        result = cache.store(
            query_original="q",
            query_rewritten=None,
            query_embedding=_embedding(),
            answer="a",
            citations=[],
            model_used="m",
        )
        assert result == ""

    def test_returns_cache_key(self) -> None:
        engine = FakeEngine()
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        result = cache.store(
            query_original="test query",
            query_rewritten=None,
            query_embedding=_embedding(),
            answer="test answer",
            citations=[{"sourceId": "c1"}],
            model_used="nova-lite",
            source_doc_ids=["doc1"],
        )
        assert len(result) == 64

    def test_store_with_source_doc_ids(self) -> None:
        engine = FakeEngine()
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        key = cache.store(
            query_original="q",
            query_rewritten="rw",
            query_embedding=_embedding(),
            answer="a",
            citations=[],
            model_used="m",
            source_doc_ids=["d1", "d2"],
        )
        assert key


class TestQueryCacheInvalidate:
    def test_invalidate_by_doc(self) -> None:
        engine = FakeEngine()
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        deleted = cache.invalidate_by_doc("doc-123")
        assert deleted == 0


class TestQueryCacheCleanup:
    def test_cleanup_expired(self) -> None:
        engine = FakeEngine()
        cache = QueryCache(settings=_settings(), engine=engine)
        cache._table_ensured = True
        deleted = cache.cleanup_expired()
        assert deleted == 0
