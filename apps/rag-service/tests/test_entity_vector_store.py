"""Tests for entity_vector_store.py — EntityVectorStore and helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.entity_vector_store import EntityVectorStore, _to_vector_literal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides):
    mock = MagicMock()
    mock.db_host = "localhost"
    mock.db_port = 5432
    mock.db_name = "testdb"
    mock.db_user = "testuser"
    mock.db_password = "testpass"
    mock.db_password_secret_arn = ""
    mock.build_db_dsn.return_value = "postgresql://testuser:testpass@localhost:5432/testdb"
    for k, v in overrides.items():
        setattr(mock, k, v)
    return mock


def _make_mock_engine():
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine, conn


def _sample_embedding(dim: int = 1024) -> list[float]:
    return [0.1] * dim


# ---------------------------------------------------------------------------
# _to_vector_literal
# ---------------------------------------------------------------------------


class TestToVectorLiteral:
    def test_basic_conversion(self):
        result = _to_vector_literal([0.1, 0.2, 0.3])
        assert result.startswith("[")
        assert result.endswith("]")
        assert "0.100000000" in result
        assert "0.200000000" in result

    def test_empty_list(self):
        result = _to_vector_literal([])
        assert result == "[]"

    def test_precision(self):
        result = _to_vector_literal([1.123456789012])
        assert "1.123456789" in result


# ---------------------------------------------------------------------------
# EntityVectorStore init and engine
# ---------------------------------------------------------------------------


class TestEntityVectorStoreInit:
    def test_creates_with_settings(self):
        settings = _make_settings()
        store = EntityVectorStore(settings)
        assert store._engine is None

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_lazy_engine_creation(self, mock_create_engine, mock_resolve):
        settings = _make_settings()
        store = EntityVectorStore(settings)
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        engine = store._get_engine()
        assert engine is mock_engine
        mock_create_engine.assert_called_once()

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_engine_cached(self, mock_create_engine, mock_resolve):
        settings = _make_settings()
        store = EntityVectorStore(settings)
        mock_create_engine.return_value = MagicMock()

        store._get_engine()
        store._get_engine()
        mock_create_engine.assert_called_once()

    @patch("app.entity_vector_store.resolve_db_password", return_value="")
    def test_no_password_raises(self, mock_resolve):
        settings = _make_settings()
        store = EntityVectorStore(settings)
        with pytest.raises(ValueError, match="DB password is not configured"):
            store._get_engine()


# ---------------------------------------------------------------------------
# upsert_entity
# ---------------------------------------------------------------------------


class TestUpsertEntity:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_entity_id(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("ent-001",)

        store = EntityVectorStore(_make_settings())
        result = store.upsert_entity(
            entity_id="ent-001",
            name="Test Entity",
            entity_type="Work",
            description="A test entity",
            embedding=_sample_embedding(),
        )
        assert result == "ent-001"

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_none_on_exception(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.side_effect = RuntimeError("DB error")

        store = EntityVectorStore(_make_settings())
        result = store.upsert_entity(
            entity_id="ent-001",
            name="Test",
            entity_type="Work",
            description="desc",
            embedding=_sample_embedding(),
        )
        assert result is None

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_passes_optional_fields(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("ent-001",)

        store = EntityVectorStore(_make_settings())
        store.upsert_entity(
            entity_id="ent-001",
            name="Test",
            entity_type="Work",
            description="desc",
            embedding=_sample_embedding(),
            canonical_key="test_key",
            aliases=["alias1", "alias2"],
            confidence=0.95,
            source_chunk_ids=["chunk-1"],
            metadata={"custom": "value"},
        )
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["canonical_key"] == "test_key"
        assert params["aliases"] == ["alias1", "alias2"]
        assert params["confidence"] == 0.95
        assert params["source_chunk_ids"] == ["chunk-1"]

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_defaults_optional_fields(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("ent-001",)

        store = EntityVectorStore(_make_settings())
        store.upsert_entity(
            entity_id="ent-001",
            name="Test",
            entity_type="Work",
            description="desc",
            embedding=_sample_embedding(),
        )
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["canonical_key"] == ""
        assert params["aliases"] == []
        assert params["source_chunk_ids"] == []


# ---------------------------------------------------------------------------
# upsert_relation
# ---------------------------------------------------------------------------


class TestUpsertRelation:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_relation_id(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("rel-001",)

        store = EntityVectorStore(_make_settings())
        result = store.upsert_relation(
            relation_id="rel-001",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relation_type="WROTE",
            evidence="John wrote the song",
            embedding=_sample_embedding(),
        )
        assert result == "rel-001"

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_none_on_exception(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.side_effect = RuntimeError("DB error")

        store = EntityVectorStore(_make_settings())
        result = store.upsert_relation(
            relation_id="rel-001",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relation_type="WROTE",
            evidence="evidence",
            embedding=_sample_embedding(),
        )
        assert result is None

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_passes_all_params(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("rel-001",)

        store = EntityVectorStore(_make_settings())
        store.upsert_relation(
            relation_id="rel-001",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            relation_type="WROTE",
            evidence="evidence text",
            embedding=_sample_embedding(),
            confidence=0.9,
            weight=2.0,
            source_chunk_ids=["chunk-1"],
            metadata={"key": "val"},
        )
        call_args = conn.execute.call_args
        params = call_args[0][1]
        assert params["type"] == "WROTE"
        assert params["confidence"] == 0.9
        assert params["weight"] == 2.0
        assert params["source_chunk_ids"] == ["chunk-1"]


# ---------------------------------------------------------------------------
# search_entities / search_relations
# ---------------------------------------------------------------------------


class TestSearchEntities:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_results(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = [
            ("ent-001", "Entity A", "Work", "key_a", "desc A", ["alias"], 0.9, ["c1"], 0.15),
        ]

        store = EntityVectorStore(_make_settings())
        results = store.search_entities(_sample_embedding(), top_k=5)
        assert len(results) == 1
        assert results[0]["entity_id"] == "ent-001"
        assert results[0]["distance"] == 0.15
        assert results[0]["aliases"] == ["alias"]

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_empty_results(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = []

        store = EntityVectorStore(_make_settings())
        results = store.search_entities(_sample_embedding())
        assert results == []

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_handles_null_arrays(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = [
            ("ent-001", "Entity A", "Work", None, "", None, 0.0, None, 0.5),
        ]

        store = EntityVectorStore(_make_settings())
        results = store.search_entities(_sample_embedding())
        assert results[0]["aliases"] == []
        assert results[0]["source_chunk_ids"] == []
        assert results[0]["canonical_key"] is None


class TestSearchRelations:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_results(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = [
            (
                "rel-001",
                "ent-001",
                "ent-002",
                "WROTE",
                "evidence",
                0.9,
                1.0,
                ["c1"],
                0.2,
                "Entity A",
                "Person",
                "Entity B",
                "Work",
            ),
        ]

        store = EntityVectorStore(_make_settings())
        results = store.search_relations(_sample_embedding(), top_k=5)
        assert len(results) == 1
        assert results[0]["relation_id"] == "rel-001"
        assert results[0]["distance"] == 0.2
        assert results[0]["type"] == "WROTE"
        assert results[0]["source_name"] == "Entity A"
        assert results[0]["target_name"] == "Entity B"

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_empty_results(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = []

        store = EntityVectorStore(_make_settings())
        results = store.search_relations(_sample_embedding())
        assert results == []


# ---------------------------------------------------------------------------
# get_entity / get_entities_by_name
# ---------------------------------------------------------------------------


class TestGetEntity:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_found(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = (
            "ent-001",
            "Entity A",
            "Work",
            "key_a",
            "desc A",
            ["alias"],
            0.9,
            ["c1"],
        )

        store = EntityVectorStore(_make_settings())
        result = store.get_entity("ent-001")
        assert result is not None
        assert result["entity_id"] == "ent-001"
        assert result["name"] == "Entity A"

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_not_found(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = None

        store = EntityVectorStore(_make_settings())
        result = store.get_entity("nonexistent")
        assert result is None


class TestGetEntitiesByName:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_found_multiple(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = [
            ("ent-001", "Test", "Work", None, "desc1", [], 0.8, []),
            ("ent-002", "Test", "Person", None, "desc2", [], 0.9, []),
        ]

        store = EntityVectorStore(_make_settings())
        results = store.get_entities_by_name("Test")
        assert len(results) == 2

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_not_found(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = []

        store = EntityVectorStore(_make_settings())
        results = store.get_entities_by_name("nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# delete_entities_by_source_chunks
# ---------------------------------------------------------------------------


class TestDeleteEntitiesBySourceChunks:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_returns_deleted_ids(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchall.return_value = [("ent-001",), ("ent-002",)]

        store = EntityVectorStore(_make_settings())
        result = store.delete_entities_by_source_chunks(["chunk-1", "chunk-2"])
        assert result == ["ent-001", "ent-002"]

    def test_empty_input_returns_empty(self):
        store = EntityVectorStore(_make_settings())
        result = store.delete_entities_by_source_chunks([])
        assert result == []


# ---------------------------------------------------------------------------
# batch_upsert_entities / batch_upsert_relations
# ---------------------------------------------------------------------------


class TestBatchUpsertEntities:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_upserts_all(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("ent-001",)

        store = EntityVectorStore(_make_settings())
        entities = [
            {
                "entity_id": "ent-001",
                "name": "A",
                "type": "Work",
                "description": "d",
                "embedding": _sample_embedding(),
            },
            {
                "entity_id": "ent-002",
                "name": "B",
                "type": "Person",
                "description": "d",
                "embedding": _sample_embedding(),
            },
        ]
        count = store.batch_upsert_entities(entities)
        assert count == 2

    def test_empty_input(self):
        store = EntityVectorStore(_make_settings())
        assert store.batch_upsert_entities([]) == 0

    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_counts_failures(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.side_effect = [("ent-001",), RuntimeError("fail")]

        store = EntityVectorStore(_make_settings())
        entities = [
            {
                "entity_id": "ent-001",
                "name": "A",
                "type": "Work",
                "description": "d",
                "embedding": _sample_embedding(),
            },
            {
                "entity_id": "ent-002",
                "name": "B",
                "type": "Work",
                "description": "d",
                "embedding": _sample_embedding(),
            },
        ]
        count = store.batch_upsert_entities(entities)
        assert count >= 1


class TestBatchUpsertRelations:
    @patch("app.entity_vector_store.resolve_db_password", return_value="testpass")
    @patch("app.entity_vector_store.create_engine")
    def test_upserts_all(self, mock_create_engine, mock_resolve):
        engine, conn = _make_mock_engine()
        mock_create_engine.return_value = engine
        conn.execute.return_value.fetchone.return_value = ("rel-001",)

        store = EntityVectorStore(_make_settings())
        relations = [
            {
                "relation_id": "rel-001",
                "source_entity_id": "ent-001",
                "target_entity_id": "ent-002",
                "type": "WROTE",
                "evidence": "ev",
                "embedding": _sample_embedding(),
            },
        ]
        count = store.batch_upsert_relations(relations)
        assert count == 1

    def test_empty_input(self):
        store = EntityVectorStore(_make_settings())
        assert store.batch_upsert_relations([]) == 0
