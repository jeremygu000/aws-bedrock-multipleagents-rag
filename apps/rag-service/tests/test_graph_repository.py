from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.entity_extraction_models import (
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    RelationType,
)
from app.graph_repository import (
    Neo4jRepository,
    _entity_to_params,
    _node_to_entity,
    _relation_to_params,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_entity(
    entity_id: str = "e1",
    name: str = "Yesterday",
    entity_type: EntityType = EntityType.WORK,
    canonical_key: str | None = "yesterday",
    description: str = "A famous song",
    aliases: list[str] | None = None,
    confidence: float = 0.9,
    source_chunk_ids: list[str] | None = None,
) -> ExtractedEntity:
    return ExtractedEntity(
        entity_id=entity_id,
        type=entity_type,
        name=name,
        canonical_key=canonical_key,
        description=description,
        aliases=aliases or [],
        confidence=confidence,
        source_chunk_ids=source_chunk_ids or [],
    )


def _make_relation(
    rel_type: RelationType = RelationType.WROTE,
    source_entity_id: str = "e1",
    target_entity_id: str = "e2",
    evidence: str = "wrote the song",
    confidence: float = 0.85,
    weight: float = 1.0,
    source_chunk_ids: list[str] | None = None,
) -> ExtractedRelation:
    return ExtractedRelation.model_construct(
        type=rel_type,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        evidence=evidence,
        confidence=confidence,
        weight=weight,
        source_chunk_ids=source_chunk_ids or [],
    )


def _make_repo(use_apoc: bool = False) -> Neo4jRepository:
    return Neo4jRepository(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test",
        database="testdb",
        use_apoc=use_apoc,
    )


def _mock_driver():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, session


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestEntityToParams:
    def test_basic_conversion(self):
        entity = _make_entity()
        params = _entity_to_params(entity)
        assert params["entity_id"] == "e1"
        assert params["name"] == "Yesterday"
        assert params["type"] == "Work"
        assert params["canonical_key"] == "yesterday"
        assert params["description"] == "A famous song"
        assert params["aliases"] == []
        assert params["confidence"] == 0.9
        assert params["source_chunk_ids"] == []

    def test_none_canonical_key_becomes_empty_string(self):
        entity = _make_entity(canonical_key=None)
        params = _entity_to_params(entity)
        assert params["canonical_key"] == ""

    def test_aliases_and_chunk_ids_preserved(self):
        entity = _make_entity(aliases=["Y-day"], source_chunk_ids=["c1", "c2"])
        params = _entity_to_params(entity)
        assert params["aliases"] == ["Y-day"]
        assert params["source_chunk_ids"] == ["c1", "c2"]


class TestRelationToParams:
    def test_basic_conversion(self):
        src = _make_entity(entity_id="e1", name="John", entity_type=EntityType.PERSON)
        tgt = _make_entity(entity_id="e2", name="Yesterday", entity_type=EntityType.WORK)
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2")
        entity_map = {"e1": src, "e2": tgt}
        params = _relation_to_params(rel, entity_map)
        assert params is not None
        assert params["source_name"] == "John"
        assert params["source_type"] == "Person"
        assert params["target_name"] == "Yesterday"
        assert params["target_type"] == "Work"
        assert params["rel_type"] == "WROTE"
        assert params["evidence"] == "wrote the song"
        assert params["confidence"] == 0.85
        assert params["weight"] == 1.0

    def test_missing_source_returns_none(self):
        tgt = _make_entity(entity_id="e2")
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2")
        params = _relation_to_params(rel, {"e2": tgt})
        assert params is None

    def test_missing_target_returns_none(self):
        src = _make_entity(entity_id="e1")
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2")
        params = _relation_to_params(rel, {"e1": src})
        assert params is None

    def test_empty_entity_map_returns_none(self):
        rel = _make_relation()
        params = _relation_to_params(rel, {})
        assert params is None


class TestNodeToEntity:
    def test_from_dict(self):
        node = {
            "entity_id": "e1",
            "type": "Person",
            "name": "John",
            "canonical_key": "john",
            "description": "A person",
            "aliases": ["Johnny"],
            "confidence": 0.95,
            "source_chunk_ids": ["c1"],
        }
        entity = _node_to_entity(node)
        assert entity.entity_id == "e1"
        assert entity.type == EntityType.PERSON
        assert entity.name == "John"
        assert entity.canonical_key == "john"
        assert entity.description == "A person"
        assert entity.aliases == ["Johnny"]
        assert entity.confidence == 0.95
        assert entity.source_chunk_ids == ["c1"]

    def test_missing_fields_use_defaults(self):
        node = {"name": "Test", "type": "Work"}
        entity = _node_to_entity(node)
        assert entity.entity_id == ""
        assert entity.description == ""
        assert entity.aliases == []
        assert entity.confidence == 0.0
        assert entity.source_chunk_ids == []

    def test_none_canonical_key(self):
        node = {"name": "Test", "type": "Work", "canonical_key": None}
        entity = _node_to_entity(node)
        assert entity.canonical_key is None

    def test_empty_canonical_key(self):
        node = {"name": "Test", "type": "Work", "canonical_key": ""}
        entity = _node_to_entity(node)
        assert entity.canonical_key is None


# ---------------------------------------------------------------------------
# Neo4jRepository tests (mocked driver)
# ---------------------------------------------------------------------------


class TestNeo4jRepositoryInit:
    def test_lazy_driver_not_created_on_init(self):
        repo = _make_repo()
        assert repo._driver is None

    def test_constructor_stores_params(self):
        repo = _make_repo()
        assert repo._uri == "bolt://localhost:7687"
        assert repo._username == "neo4j"
        assert repo._password == "test"
        assert repo._database == "testdb"
        assert repo._use_apoc is False


class TestNeo4jRepositoryConnection:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_get_driver_creates_on_first_call(self, mock_gd_driver):
        mock_driver = MagicMock()
        mock_gd_driver.return_value = mock_driver
        repo = _make_repo()
        driver = repo._get_driver()
        assert driver is mock_driver
        mock_gd_driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "test"),
        )

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_get_driver_reuses_existing(self, mock_gd_driver):
        mock_driver = MagicMock()
        mock_gd_driver.return_value = mock_driver
        repo = _make_repo()
        repo._get_driver()
        repo._get_driver()
        mock_gd_driver.assert_called_once()

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_close_shuts_down_driver(self, mock_gd_driver):
        mock_driver = MagicMock()
        mock_gd_driver.return_value = mock_driver
        repo = _make_repo()
        repo._get_driver()
        repo.close()
        mock_driver.close.assert_called_once()
        assert repo._driver is None

    def test_close_noop_when_no_driver(self):
        repo = _make_repo()
        repo.close()
        assert repo._driver is None


class TestUpsertEntity:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_upsert_entity_returns_entity_id(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        mock_record = {"entity_id": "e1"}
        mock_session.execute_write.return_value = "e1"

        repo = _make_repo()
        entity = _make_entity()

        # We need to intercept the _work function passed to execute_write
        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.single.return_value = mock_record
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect
        result = repo.upsert_entity(entity)
        assert result == "e1"

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_upsert_entity_returns_none_on_no_record(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.single.return_value = None
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect
        repo = _make_repo()
        result = repo.upsert_entity(_make_entity())
        assert result is None

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_upsert_entity_uses_no_apoc_cypher_by_default(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver
        captured_cypher = []

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.single.return_value = {"entity_id": "e1"}
            tx.run.return_value = result

            def capture_run(cypher, **kwargs):
                captured_cypher.append(cypher)
                return result

            tx.run = capture_run
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect
        repo = _make_repo(use_apoc=False)
        repo.upsert_entity(_make_entity())
        assert len(captured_cypher) == 1
        assert "apoc" not in captured_cypher[0].lower()

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_upsert_entity_uses_apoc_cypher_when_enabled(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver
        captured_cypher = []

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.single.return_value = {"entity_id": "e1"}
            tx.run.return_value = result

            def capture_run(cypher, **kwargs):
                captured_cypher.append(cypher)
                return result

            tx.run = capture_run
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect
        repo = _make_repo(use_apoc=True)
        repo._use_apoc = True
        repo.upsert_entity(_make_entity())
        assert len(captured_cypher) == 1
        assert "apoc" in captured_cypher[0].lower()


class TestUpsertRelation:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_upsert_relation_returns_true(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.single.return_value = {"rel": "RELATES_TO"}
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect

        src = _make_entity(entity_id="e1", name="John", entity_type=EntityType.PERSON)
        tgt = _make_entity(entity_id="e2", name="Yesterday", entity_type=EntityType.WORK)
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2")
        entity_map = {"e1": src, "e2": tgt}

        repo = _make_repo()
        result = repo.upsert_relation(rel, entity_map)
        assert result is True

    def test_upsert_relation_returns_false_for_missing_entity(self):
        repo = _make_repo()
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2")
        result = repo.upsert_relation(rel, {})
        assert result is False


class TestUpsertBatch:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_batch_upsert_counts(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 2
        mock_summary.counters.properties_set = 10
        mock_summary.counters.relationships_created = 1

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.consume.return_value = mock_summary
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect

        src = _make_entity(entity_id="e1", name="John", entity_type=EntityType.PERSON)
        tgt = _make_entity(entity_id="e2", name="Yesterday", entity_type=EntityType.WORK)
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2")

        repo = _make_repo()
        result = repo.upsert_batch([src, tgt], [rel])
        assert result["entities_written"] == 2
        assert result["relations_written"] == 1

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_batch_upsert_empty_lists(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        repo = _make_repo()
        result = repo.upsert_batch([], [])
        assert result == {"entities_written": 0, "relations_written": 0}
        mock_session.execute_write.assert_not_called()

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_batch_upsert_skips_invalid_relations(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        mock_summary = MagicMock()
        mock_summary.counters.nodes_created = 1
        mock_summary.counters.properties_set = 5
        mock_summary.counters.relationships_created = 0

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.consume.return_value = mock_summary
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_write.side_effect = side_effect

        entity = _make_entity(entity_id="e1")
        # Relation points to non-existent e99
        rel = _make_relation(source_entity_id="e1", target_entity_id="e99")

        repo = _make_repo()
        result = repo.upsert_batch([entity], [rel])
        assert result["entities_written"] == 1
        assert result["relations_written"] == 0


class TestGetEntity:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_get_entity_found(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        node_data = {
            "entity_id": "e1",
            "type": "Work",
            "name": "Yesterday",
            "canonical_key": "yesterday",
            "description": "A song",
            "aliases": [],
            "confidence": 0.9,
            "source_chunk_ids": ["c1"],
        }

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            record = MagicMock()
            record.__getitem__ = lambda self, key: node_data
            result.single.return_value = record
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        repo = _make_repo()
        entity = repo.get_entity("Yesterday", EntityType.WORK)
        assert entity is not None
        assert entity.name == "Yesterday"
        assert entity.type == EntityType.WORK

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_get_entity_not_found(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.single.return_value = None
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        repo = _make_repo()
        entity = repo.get_entity("NonExistent", EntityType.WORK)
        assert entity is None


class TestGetEntityNeighbors:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_get_neighbors_no_apoc(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        neighbor_data = {
            "entity_id": "e2",
            "type": "Person",
            "name": "John",
            "canonical_key": "john",
            "description": "",
            "aliases": [],
            "confidence": 0.8,
            "source_chunk_ids": [],
        }

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            record = MagicMock()
            record.__getitem__ = lambda self, key: neighbor_data
            result.__iter__ = lambda self: iter([record])
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        repo = _make_repo(use_apoc=False)
        neighbors = repo.get_entity_neighbors("Yesterday", EntityType.WORK, depth=2)
        assert len(neighbors) == 1
        assert neighbors[0].name == "John"

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_get_neighbors_empty(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            result.__iter__ = lambda self: iter([])
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        repo = _make_repo()
        neighbors = repo.get_entity_neighbors("Yesterday", EntityType.WORK)
        assert neighbors == []


class TestGetRelationsForEntities:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_returns_relation_dicts(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        rel_record = {
            "source_name": "John",
            "source_type": "Person",
            "target_name": "Yesterday",
            "target_type": "Work",
            "rel_type": "WROTE",
            "evidence": "wrote the song",
            "confidence": 0.85,
            "weight": 1.0,
            "source_chunk_ids": ["c1"],
        }

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            record = MagicMock()
            record.keys.return_value = list(rel_record.keys())
            record.__iter__ = lambda self: iter(rel_record.items())
            result.__iter__ = lambda self: iter([record])
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        repo = _make_repo()
        rels = repo.get_relations_for_entities(["John", "Yesterday"])
        assert len(rels) == 1


class TestSearchEntitiesFulltext:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_returns_entity_score_tuples(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        node_data = {
            "entity_id": "e1",
            "type": "Work",
            "name": "Yesterday",
            "canonical_key": "yesterday",
            "description": "A song",
            "aliases": [],
            "confidence": 0.9,
            "source_chunk_ids": [],
        }

        def side_effect(work_fn):
            tx = MagicMock()
            result = MagicMock()
            record = MagicMock()
            record.__getitem__ = lambda self, key: node_data if key == "node" else 0.95
            result.__iter__ = lambda self: iter([record])
            tx.run.return_value = result
            return work_fn(tx)

        mock_session.execute_read.side_effect = side_effect

        repo = _make_repo()
        results = repo.search_entities_fulltext("Yesterday", limit=5)
        assert len(results) == 1
        entity, score = results[0]
        assert entity.name == "Yesterday"
        assert score == 0.95


class TestEnsureIndexes:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_runs_three_index_statements(self, mock_gd_driver):
        mock_driver, mock_session = _mock_driver()
        mock_gd_driver.return_value = mock_driver

        repo = _make_repo()
        repo.ensure_indexes()
        assert mock_session.run.call_count == 3


class TestHealthCheck:
    @patch("app.graph_repository.GraphDatabase.driver")
    def test_healthy(self, mock_gd_driver):
        mock_driver = MagicMock()
        mock_gd_driver.return_value = mock_driver
        mock_driver.verify_connectivity.return_value = None

        repo = _make_repo()
        assert repo.health_check() is True
        mock_driver.verify_connectivity.assert_called_once()

    @patch("app.graph_repository.GraphDatabase.driver")
    def test_unhealthy(self, mock_gd_driver):
        mock_driver = MagicMock()
        mock_gd_driver.return_value = mock_driver
        mock_driver.verify_connectivity.side_effect = ConnectionError("refused")

        repo = _make_repo()
        assert repo.health_check() is False


# ---------------------------------------------------------------------------
# Config / secrets tests for Neo4j
# ---------------------------------------------------------------------------


class TestNeo4jConfig:
    def test_default_neo4j_settings(self):
        from app.config import Settings

        settings = Settings()
        assert settings.enable_neo4j is False
        assert settings.neo4j_uri == "bolt://localhost:7687"
        assert settings.neo4j_username == "neo4j"
        assert settings.neo4j_password == ""
        assert settings.neo4j_password_secret_arn == ""
        assert settings.neo4j_database == "neo4j"

    def test_neo4j_settings_from_env(self, monkeypatch):
        from app.config import Settings

        monkeypatch.setenv("RAG_ENABLE_NEO4J", "true")
        monkeypatch.setenv("RAG_NEO4J_URI", "bolt://prod:7687")
        monkeypatch.setenv("RAG_NEO4J_USERNAME", "admin")
        monkeypatch.setenv("RAG_NEO4J_PASSWORD", "secret123")
        monkeypatch.setenv("RAG_NEO4J_DATABASE", "production")
        settings = Settings()
        assert settings.enable_neo4j is True
        assert settings.neo4j_uri == "bolt://prod:7687"
        assert settings.neo4j_username == "admin"
        assert settings.neo4j_password == "secret123"
        assert settings.neo4j_database == "production"


class TestResolveNeo4jPassword:
    def test_prefers_env_password(self):
        from app.config import Settings
        from app.secrets import resolve_neo4j_password

        settings = Settings(
            RAG_NEO4J_PASSWORD="neo4j-pass-from-env",
            RAG_NEO4J_PASSWORD_SECRET_ARN="arn:aws:secretsmanager:region:acct:secret:ignored",
        )
        assert resolve_neo4j_password(settings) == "neo4j-pass-from-env"

    def test_returns_empty_when_no_password_or_arn(self):
        from app.config import Settings
        from app.secrets import resolve_neo4j_password

        settings = Settings()
        assert resolve_neo4j_password(settings) == ""

    def test_resolves_from_secrets_manager(self, monkeypatch):
        from app import secrets
        from app.config import Settings
        from app.secrets import resolve_neo4j_password

        settings = Settings(
            RAG_NEO4J_PASSWORD_SECRET_ARN="arn:aws:secretsmanager:ap-southeast-2:123:secret:neo4j",
            RAG_AWS_REGION="ap-southeast-2",
        )

        def fake_fetch(secret_arn: str, region: str | None) -> str:
            assert secret_arn == settings.neo4j_password_secret_arn
            assert region == "ap-southeast-2"
            return '{"username": "neo4j", "password": "from-secret-manager"}'

        monkeypatch.setattr(secrets, "_fetch_secret_string", fake_fetch)
        assert resolve_neo4j_password(settings) == "from-secret-manager"
