"""Tests for app.graph_retriever — GraphRetriever (Phase 3.1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.entity_extraction_models import EntityType, ExtractedEntity
from app.graph_retriever import (
    GraphRetriever,
    _deduplicate_relations,
    _entity_dict_to_model,
    _merge_contexts,
    _merge_entity_dicts,
    _relation_dict_to_model,
)
from app.models import GraphContext, GraphEntity, GraphRelation, RetrievalMode

# ---------------------------------------------------------------------------
# Fixtures & factories
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 1024


def _make_settings(**overrides) -> MagicMock:
    """Build a mock Settings with graph retrieval defaults."""
    defaults = {
        "retrieval_mode": "mix",
        "graph_top_k_entities": 10,
        "graph_top_k_relations": 10,
        "graph_neighbor_depth": 1,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _make_entity_dict(
    entity_id: str = "e1",
    name: str = "Yesterday",
    entity_type: str = "Work",
    description: str = "A famous song",
    distance: float = 0.2,
    confidence: float = 0.9,
    source_chunk_ids: list[str] | None = None,
) -> dict:
    return {
        "entity_id": entity_id,
        "name": name,
        "type": entity_type,
        "canonical_key": name.lower(),
        "description": description,
        "aliases": [],
        "confidence": confidence,
        "source_chunk_ids": source_chunk_ids or [],
        "distance": distance,
    }


def _make_relation_dict(
    relation_id: str = "r1",
    source_entity_id: str = "e1",
    target_entity_id: str = "e2",
    rel_type: str = "WROTE",
    evidence: str = "wrote the song",
    distance: float = 0.3,
    confidence: float = 0.85,
    weight: float = 1.0,
    source_chunk_ids: list[str] | None = None,
) -> dict:
    return {
        "relation_id": relation_id,
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "type": rel_type,
        "evidence": evidence,
        "confidence": confidence,
        "weight": weight,
        "source_chunk_ids": source_chunk_ids or [],
        "distance": distance,
    }


def _make_neo4j_relation_dict(
    source_name: str = "Yesterday",
    source_type: str = "Work",
    target_name: str = "Beatles",
    target_type: str = "Organization",
    rel_type: str = "PERFORMED_BY",
    evidence: str = "performed by the Beatles",
    confidence: float = 0.8,
    weight: float = 1.0,
) -> dict:
    return {
        "source_name": source_name,
        "source_type": source_type,
        "target_name": target_name,
        "target_type": target_type,
        "rel_type": rel_type,
        "evidence": evidence,
        "confidence": confidence,
        "weight": weight,
        "source_chunk_ids": [],
    }


def _build_retriever(
    settings: MagicMock | None = None,
    qwen: MagicMock | None = None,
    vector_store: MagicMock | None = None,
    neo4j_repo: MagicMock | None = None,
) -> GraphRetriever:
    if vector_store is None:
        vs = MagicMock()
        vs.search_entities_by_text.return_value = []
    else:
        vs = vector_store
    return GraphRetriever(
        qwen_client=qwen or MagicMock(),
        vector_store=vs,
        neo4j_repo=neo4j_repo,
        settings=settings or _make_settings(),
    )


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


class TestEntityDictToModel:
    """Tests for _entity_dict_to_model conversion."""

    def test_converts_basic_entity(self):
        ed = _make_entity_dict(distance=0.2)
        result = _entity_dict_to_model(ed)
        assert isinstance(result, GraphEntity)
        assert result.entity_id == "e1"
        assert result.name == "Yesterday"
        assert result.type == "Work"
        assert result.description == "A famous song"
        assert result.confidence == 0.9
        assert result.score == pytest.approx(0.8)  # 1.0 - 0.2

    def test_distance_zero_gives_score_one(self):
        ed = _make_entity_dict(distance=0.0)
        result = _entity_dict_to_model(ed)
        assert result.score == pytest.approx(1.0)

    def test_distance_above_one_gives_score_zero(self):
        ed = _make_entity_dict(distance=1.5)
        result = _entity_dict_to_model(ed)
        assert result.score == 0.0

    def test_missing_distance_defaults_to_zero_score(self):
        ed = _make_entity_dict()
        del ed["distance"]
        result = _entity_dict_to_model(ed)
        assert result.score == 0.0  # default distance=1.0 → score=0.0


class TestRelationDictToModel:
    """Tests for _relation_dict_to_model conversion."""

    def test_converts_basic_relation(self):
        rd = _make_relation_dict(distance=0.3)
        result = _relation_dict_to_model(rd)
        assert isinstance(result, GraphRelation)
        assert result.source_entity == "e1"
        assert result.target_entity == "e2"
        assert result.relation_type == "WROTE"
        assert result.evidence == "wrote the song"
        assert result.score == pytest.approx(0.7)  # 1.0 - 0.3


class TestDeduplicateRelations:
    """Tests for _deduplicate_relations."""

    def test_empty_list(self):
        assert _deduplicate_relations([]) == []

    def test_no_duplicates(self):
        r1 = GraphRelation(
            source_entity="a", target_entity="b", relation_type="X", evidence="ev1", score=0.5
        )
        r2 = GraphRelation(
            source_entity="c", target_entity="d", relation_type="Y", evidence="ev2", score=0.6
        )
        result = _deduplicate_relations([r1, r2])
        assert len(result) == 2

    def test_keeps_higher_scored_duplicate(self):
        r_low = GraphRelation(
            source_entity="a", target_entity="b", relation_type="X", evidence="low", score=0.3
        )
        r_high = GraphRelation(
            source_entity="a", target_entity="b", relation_type="X", evidence="high", score=0.9
        )
        result = _deduplicate_relations([r_low, r_high])
        assert len(result) == 1
        assert result[0].evidence == "high"


class TestMergeContexts:
    """Tests for _merge_contexts."""

    def test_merge_empty_contexts(self):
        ctx = _merge_contexts(GraphContext(), GraphContext())
        assert ctx.is_empty

    def test_merge_deduplicates_entities(self):
        e1 = GraphEntity(entity_id="e1", name="A", type="Work", description="d", score=0.5)
        e1_better = GraphEntity(entity_id="e1", name="A", type="Work", description="d", score=0.9)
        e2 = GraphEntity(entity_id="e2", name="B", type="Person", description="d", score=0.4)

        a = GraphContext(entities=[e1, e2])
        b = GraphContext(entities=[e1_better])

        result = _merge_contexts(a, b)
        assert len(result.entities) == 2
        # The higher-scored version should win
        matched = [e for e in result.entities if e.entity_id == "e1"]
        assert matched[0].score == pytest.approx(0.9)

    def test_merge_deduplicates_relations(self):
        r1 = GraphRelation(
            source_entity="a", target_entity="b", relation_type="X", evidence="ev1", score=0.3
        )
        r1_better = GraphRelation(
            source_entity="a", target_entity="b", relation_type="X", evidence="ev2", score=0.8
        )
        a = GraphContext(relations=[r1])
        b = GraphContext(relations=[r1_better])
        result = _merge_contexts(a, b)
        assert len(result.relations) == 1
        assert result.relations[0].evidence == "ev2"

    def test_merge_combines_source_chunk_ids(self):
        a = GraphContext(source_chunk_ids=["c1", "c2"])
        b = GraphContext(source_chunk_ids=["c2", "c3"])
        result = _merge_contexts(a, b)
        assert result.source_chunk_ids == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Integration tests — GraphRetriever.retrieve()
# ---------------------------------------------------------------------------


class TestGraphRetrieverRetrieve:
    """Tests for GraphRetriever.retrieve() — main entry point."""

    def test_chunks_only_mode_returns_empty(self):
        """When retrieval_mode=chunks_only, skip graph entirely."""
        settings = _make_settings(retrieval_mode="chunks_only")
        retriever = _build_retriever(settings=settings)
        ctx = retriever.retrieve("who wrote yesterday")
        assert ctx.is_empty

    def test_embedding_failure_returns_empty(self):
        """If query embedding fails, return empty context gracefully."""
        qwen = MagicMock()
        qwen.embedding.side_effect = RuntimeError("API down")
        retriever = _build_retriever(qwen=qwen)
        ctx = retriever.retrieve("test query")
        assert ctx.is_empty

    def test_embedding_returns_empty_list(self):
        """If embedding returns empty result, return empty context."""
        qwen = MagicMock()
        qwen.embedding.return_value = []
        retriever = _build_retriever(qwen=qwen)
        ctx = retriever.retrieve("test query")
        assert ctx.is_empty

    def test_local_strategy_happy_path(self):
        """Local strategy: vector search → neighbor expansion → relation collection."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING

        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]

        neo4j = MagicMock()
        # Neighbor expansion returns one new entity
        neighbor = ExtractedEntity(
            entity_id="e2",
            type=EntityType.PERSON,
            name="Lennon",
            canonical_key="lennon",
            description="John Lennon",
            aliases=[],
            confidence=0.8,
            source_chunk_ids=[],
        )
        neo4j.get_entity_neighbors.return_value = [neighbor]
        neo4j.get_relations_for_entities.return_value = [
            _make_neo4j_relation_dict(
                source_name="Yesterday",
                target_name="Lennon",
                rel_type="WROTE",
                evidence="written by Lennon",
            )
        ]

        settings = _make_settings(retrieval_mode="mix", graph_neighbor_depth=1)
        retriever = _build_retriever(
            settings=settings, qwen=qwen, vector_store=vs, neo4j_repo=neo4j
        )
        ctx = retriever.retrieve("who wrote yesterday", strategy="local")

        assert len(ctx.entities) == 2
        assert len(ctx.relations) == 1
        assert ctx.relations[0].relation_type == "WROTE"

        # Verify the calls
        vs.search_entities.assert_called_once()
        neo4j.get_entity_neighbors.assert_called_once()
        neo4j.get_relations_for_entities.assert_called_once()

    def test_local_no_seeds_returns_empty(self):
        """If vector search returns no entities, return empty context."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING
        vs = MagicMock()
        vs.search_entities.return_value = []

        retriever = _build_retriever(qwen=qwen, vector_store=vs)
        ctx = retriever.retrieve("unknown query", strategy="local")
        assert ctx.is_empty

    def test_local_without_neo4j(self):
        """Local strategy without Neo4j: only vector search, no expansion or relations."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING
        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.15),
        ]

        retriever = _build_retriever(qwen=qwen, vector_store=vs, neo4j_repo=None)
        ctx = retriever.retrieve("yesterday", strategy="local")

        assert len(ctx.entities) == 1
        assert len(ctx.relations) == 0

    def test_local_neo4j_neighbor_failure_continues(self):
        """If Neo4j neighbor expansion fails, still return seed entities."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING
        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]
        neo4j = MagicMock()
        neo4j.get_entity_neighbors.side_effect = RuntimeError("Neo4j down")
        neo4j.get_relations_for_entities.return_value = []

        retriever = _build_retriever(qwen=qwen, vector_store=vs, neo4j_repo=neo4j)
        ctx = retriever.retrieve("yesterday", strategy="local")

        # Should still have seed entity despite neighbor failure
        assert len(ctx.entities) == 1
        assert ctx.entities[0].name == "Yesterday"

    def test_local_neo4j_relation_failure_continues(self):
        """If Neo4j relation fetch fails, still return entities."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING
        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]
        neo4j = MagicMock()
        neo4j.get_entity_neighbors.return_value = []
        neo4j.get_relations_for_entities.side_effect = RuntimeError("Neo4j down")

        retriever = _build_retriever(qwen=qwen, vector_store=vs, neo4j_repo=neo4j)
        ctx = retriever.retrieve("yesterday", strategy="local")

        assert len(ctx.entities) == 1
        assert len(ctx.relations) == 0

    def test_global_strategy_happy_path(self):
        """Global strategy: vector search entities + relations directly."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING

        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]
        vs.search_relations.return_value = [
            _make_relation_dict("r1", "e1", "e2", "WROTE", "wrote it", distance=0.2),
        ]

        retriever = _build_retriever(qwen=qwen, vector_store=vs)
        ctx = retriever.retrieve("who wrote yesterday", strategy="global")

        assert len(ctx.entities) == 1
        assert len(ctx.relations) == 1

    def test_hybrid_strategy_merges_local_and_global(self):
        """Hybrid strategy should merge local and global results."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING

        vs = MagicMock()
        # search_entities called twice (local + global)
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]
        # search_relations only called in global
        vs.search_relations.return_value = [
            _make_relation_dict("r1", "e1", "e2", "WROTE", "wrote it", distance=0.2),
        ]

        neo4j = MagicMock()
        neo4j.get_entity_neighbors.return_value = []
        neo4j.get_relations_for_entities.return_value = []

        retriever = _build_retriever(qwen=qwen, vector_store=vs, neo4j_repo=neo4j)
        ctx = retriever.retrieve("who wrote yesterday", strategy="hybrid")

        # Should have entity from both local+global (deduplicated)
        assert len(ctx.entities) == 1
        # Relations from global
        assert len(ctx.relations) == 1

    def test_unknown_entity_type_skips_neighbor_expansion(self):
        """If entity type is not a valid EntityType, skip neighbor expansion for that entity."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING

        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Something", "unknown_type", distance=0.1),
        ]

        neo4j = MagicMock()
        neo4j.get_relations_for_entities.return_value = []

        retriever = _build_retriever(qwen=qwen, vector_store=vs, neo4j_repo=neo4j)
        ctx = retriever.retrieve("test", strategy="local")

        # Entity should still be present, but no neighbors expanded
        assert len(ctx.entities) == 1
        neo4j.get_entity_neighbors.assert_not_called()

    def test_depth_zero_skips_neighbor_expansion(self):
        """When graph_neighbor_depth=0, skip Neo4j neighbor expansion entirely."""
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING
        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]
        neo4j = MagicMock()
        neo4j.get_relations_for_entities.return_value = []

        settings = _make_settings(graph_neighbor_depth=0)
        retriever = _build_retriever(
            settings=settings, qwen=qwen, vector_store=vs, neo4j_repo=neo4j
        )
        ctx = retriever.retrieve("yesterday", strategy="local")

        assert len(ctx.entities) == 1
        neo4j.get_entity_neighbors.assert_not_called()

    def test_embedding_returns_nested_list(self):
        """Handle case where embedding() returns list[list[float]] instead of list[float]."""
        qwen = MagicMock()
        qwen.embedding.return_value = [FAKE_EMBEDDING]  # wrapped in outer list

        vs = MagicMock()
        vs.search_entities.return_value = [
            _make_entity_dict("e1", "Yesterday", "Work", distance=0.1),
        ]
        vs.search_relations.return_value = []

        retriever = _build_retriever(qwen=qwen, vector_store=vs)
        ctx = retriever.retrieve("yesterday", strategy="global")

        # Should still work — unwrap nested list
        assert len(ctx.entities) == 1


# ---------------------------------------------------------------------------
# GraphContext model tests
# ---------------------------------------------------------------------------


class TestGraphContextModel:
    """Tests for GraphContext Pydantic model."""

    def test_is_empty_true(self):
        ctx = GraphContext()
        assert ctx.is_empty is True

    def test_is_empty_false_with_entities(self):
        ctx = GraphContext(
            entities=[GraphEntity(entity_id="e1", name="A", type="Work", description="d")]
        )
        assert ctx.is_empty is False

    def test_is_empty_false_with_relations(self):
        ctx = GraphContext(
            relations=[
                GraphRelation(
                    source_entity="a",
                    target_entity="b",
                    relation_type="X",
                    evidence="ev",
                )
            ]
        )
        assert ctx.is_empty is False

    def test_to_evidence_text_empty(self):
        ctx = GraphContext()
        assert ctx.to_evidence_text() == ""

    def test_to_evidence_text_with_entities_and_relations(self):
        ctx = GraphContext(
            entities=[
                GraphEntity(
                    entity_id="e1",
                    name="Yesterday",
                    type="Work",
                    description="A famous Beatles song",
                    score=0.9,
                )
            ],
            relations=[
                GraphRelation(
                    source_entity="Yesterday",
                    target_entity="Beatles",
                    relation_type="PERFORMED_BY",
                    evidence="performed by the Beatles",
                    score=0.7,
                )
            ],
        )
        text = ctx.to_evidence_text()
        assert "### Entities" in text
        assert "Yesterday (Work)" in text
        assert "### Relations" in text
        assert "PERFORMED_BY" in text

    def test_to_evidence_text_respects_max_limits(self):
        """Only include up to max_entities and max_relations."""
        entities = [
            GraphEntity(
                entity_id=f"e{i}",
                name=f"Entity{i}",
                type="Work",
                description=f"desc{i}",
                score=float(i) / 10,
            )
            for i in range(5)
        ]
        ctx = GraphContext(entities=entities)
        text = ctx.to_evidence_text(max_entities=2)
        # Should only have 2 entities (highest scored)
        assert text.count("- Entity") == 2


class TestRetrievalModeEnum:
    """Tests for RetrievalMode enum."""

    def test_valid_values(self):
        assert RetrievalMode("chunks_only") == RetrievalMode.CHUNKS_ONLY
        assert RetrievalMode("graph_only") == RetrievalMode.GRAPH_ONLY
        assert RetrievalMode("mix") == RetrievalMode.MIX

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RetrievalMode("invalid")


class TestMergeEntityDicts:

    def test_empty_inputs(self):
        assert _merge_entity_dicts([], []) == []

    def test_vector_only(self):
        v = [_make_entity_dict("e1", "A", distance=0.3)]
        result = _merge_entity_dicts(v, [])
        assert len(result) == 1
        assert result[0]["entity_id"] == "e1"

    def test_text_only(self):
        t = [_make_entity_dict("e1", "A", distance=0.05)]
        result = _merge_entity_dicts([], t)
        assert len(result) == 1

    def test_text_overrides_worse_vector(self):
        v = [_make_entity_dict("e1", "A", distance=0.5)]
        t = [_make_entity_dict("e1", "A", distance=0.0)]
        result = _merge_entity_dicts(v, t)
        assert len(result) == 1
        assert result[0]["distance"] == 0.0

    def test_vector_kept_when_better(self):
        v = [_make_entity_dict("e1", "A", distance=0.01)]
        t = [_make_entity_dict("e1", "A", distance=0.15)]
        result = _merge_entity_dicts(v, t)
        assert len(result) == 1
        assert result[0]["distance"] == 0.01

    def test_union_of_different_entities(self):
        v = [_make_entity_dict("e1", "A", distance=0.2)]
        t = [_make_entity_dict("e2", "B", distance=0.05)]
        result = _merge_entity_dicts(v, t)
        assert len(result) == 2
        assert result[0]["entity_id"] == "e2"
        assert result[1]["entity_id"] == "e1"

    def test_sorted_by_distance(self):
        v = [
            _make_entity_dict("e1", "A", distance=0.5),
            _make_entity_dict("e2", "B", distance=0.1),
        ]
        t = [_make_entity_dict("e3", "C", distance=0.0)]
        result = _merge_entity_dicts(v, t)
        distances = [r["distance"] for r in result]
        assert distances == sorted(distances)


class TestTextSearchIntegration:

    def test_local_uses_text_search(self):
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING

        vs = MagicMock()
        vs.search_entities.return_value = []
        vs.search_entities_by_text.return_value = [
            _make_entity_dict("e1", "Rushing Back", "Work", distance=0.0),
        ]

        neo4j = MagicMock()
        neo4j.get_entity_neighbors.return_value = []
        neo4j.get_relations_for_entities.return_value = []

        settings = _make_settings(retrieval_mode="mix", graph_neighbor_depth=1)
        retriever = _build_retriever(
            settings=settings, qwen=qwen, vector_store=vs, neo4j_repo=neo4j
        )
        ctx = retriever.retrieve("Who wrote Rushing Back?", strategy="local")

        assert len(ctx.entities) == 1
        assert ctx.entities[0].name == "Rushing Back"
        assert ctx.entities[0].score == pytest.approx(1.0)
        vs.search_entities_by_text.assert_called_once()

    def test_global_uses_text_search(self):
        qwen = MagicMock()
        qwen.embedding.return_value = FAKE_EMBEDDING

        vs = MagicMock()
        vs.search_entities.return_value = []
        vs.search_relations.return_value = []
        vs.search_entities_by_text.return_value = [
            _make_entity_dict("e1", "APRA AMCOS", "Organization", distance=0.0),
        ]

        retriever = _build_retriever(qwen=qwen, vector_store=vs)
        ctx = retriever.retrieve("What is APRA AMCOS?", strategy="global")

        assert len(ctx.entities) == 1
        assert ctx.entities[0].name == "APRA AMCOS"
