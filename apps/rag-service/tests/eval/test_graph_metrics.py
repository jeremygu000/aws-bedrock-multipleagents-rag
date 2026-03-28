from __future__ import annotations

from app.eval_metrics import (
    cache_hit_quality,
    fusion_contribution_ratio,
    graph_traversal_depth,
    multi_source_recall,
)
from app.models import GraphContext, GraphEntity, GraphRelation


class TestGraphTraversalDepth:
    def test_no_graph_context_returns_zero(self):
        state: dict = {"graph_context": None}
        assert graph_traversal_depth(state) == 0

    def test_missing_graph_context_returns_zero(self):
        state: dict = {}
        assert graph_traversal_depth(state) == 0

    def test_empty_graph_context_returns_zero(self):
        ctx = GraphContext(entities=[], relations=[])
        assert graph_traversal_depth({"graph_context": ctx}) == 0

    def test_entities_only_returns_one(self):
        ctx = GraphContext(
            entities=[
                GraphEntity(
                    entity_id="e1",
                    name="Artist",
                    type="PERSON",
                    description="A musician",
                    score=0.9,
                )
            ],
            relations=[],
        )
        assert graph_traversal_depth({"graph_context": ctx}) == 1

    def test_with_relations_returns_two(self):
        ctx = GraphContext(
            entities=[
                GraphEntity(
                    entity_id="e1",
                    name="Artist",
                    type="PERSON",
                    description="A musician",
                    score=0.9,
                )
            ],
            relations=[
                GraphRelation(
                    source_entity="Artist",
                    target_entity="Album",
                    relation_type="CREATED",
                    evidence="Artist created Album",
                    score=0.8,
                )
            ],
        )
        assert graph_traversal_depth({"graph_context": ctx}) == 2

    def test_dict_graph_context_empty(self):
        assert graph_traversal_depth({"graph_context": {"entities": [], "relations": []}}) == 0

    def test_dict_graph_context_with_entities(self):
        ctx = {"entities": [{"name": "X"}], "relations": []}
        assert graph_traversal_depth({"graph_context": ctx}) == 1

    def test_dict_graph_context_with_relations(self):
        ctx = {"entities": [], "relations": [{"type": "REL"}]}
        assert graph_traversal_depth({"graph_context": ctx}) == 2


class TestFusionContributionRatio:
    def test_empty_hits_returns_zeros(self):
        result = fusion_contribution_ratio({"reranked_hits": []})
        assert result == {"graph": 0.0, "vector": 0.0}

    def test_no_hits_key_returns_zeros(self):
        result = fusion_contribution_ratio({})
        assert result == {"graph": 0.0, "vector": 0.0}

    def test_all_graph_hits(self):
        hits = [{"source": "graph"}, {"source": "graph"}, {"source": "graph"}]
        result = fusion_contribution_ratio({"reranked_hits": hits})
        assert result == {"graph": 1.0, "vector": 0.0}

    def test_all_vector_hits(self):
        hits = [{"source": "vector"}, {"source": "vector"}]
        result = fusion_contribution_ratio({"reranked_hits": hits})
        assert result == {"graph": 0.0, "vector": 1.0}

    def test_mixed_hits(self):
        hits = [
            {"source": "graph"},
            {"source": "vector"},
            {"source": "graph"},
            {"source": "vector"},
        ]
        result = fusion_contribution_ratio({"reranked_hits": hits})
        assert result["graph"] == 0.5
        assert result["vector"] == 0.5

    def test_hits_without_source_counted_as_vector(self):
        hits = [{"chunk_id": "c1"}, {"source": "graph"}]
        result = fusion_contribution_ratio({"reranked_hits": hits})
        assert result["graph"] == 0.5
        assert result["vector"] == 0.5


class TestMultiSourceRecall:
    def test_empty_hits(self):
        assert multi_source_recall({"reranked_hits": []}) == 0

    def test_no_hits_key(self):
        assert multi_source_recall({}) == 0

    def test_single_doc(self):
        hits = [{"doc_id": "doc_a"}, {"doc_id": "doc_a"}]
        assert multi_source_recall({"reranked_hits": hits}) == 1

    def test_multiple_docs(self):
        hits = [
            {"doc_id": "doc_a"},
            {"doc_id": "doc_b"},
            {"doc_id": "doc_c"},
            {"doc_id": "doc_a"},
        ]
        assert multi_source_recall({"reranked_hits": hits}) == 3

    def test_fallback_to_document_id_key(self):
        hits = [{"document_id": "doc_x"}, {"document_id": "doc_y"}]
        assert multi_source_recall({"reranked_hits": hits}) == 2

    def test_mixed_id_keys(self):
        hits = [{"doc_id": "doc_a"}, {"document_id": "doc_b"}]
        assert multi_source_recall({"reranked_hits": hits}) == 2

    def test_hits_without_doc_id_ignored(self):
        hits = [{"chunk_id": "c1"}, {"doc_id": "doc_a"}]
        assert multi_source_recall({"reranked_hits": hits}) == 1


class TestCacheHitQuality:
    def test_identical_answers(self):
        cached = {"answer": "The ISRC code is US1234567890"}
        fresh = {"answer": "The ISRC code is US1234567890"}
        assert cache_hit_quality(cached, fresh) == 1.0

    def test_completely_different_answers(self):
        cached = {"answer": "alpha bravo charlie"}
        fresh = {"answer": "delta echo foxtrot"}
        assert cache_hit_quality(cached, fresh) == 0.0

    def test_partial_overlap(self):
        cached = {"answer": "the artist released an album in 2020"}
        fresh = {"answer": "the artist released a single in 2021"}
        score = cache_hit_quality(cached, fresh)
        assert 0.0 < score < 1.0

    def test_empty_cached_answer(self):
        assert cache_hit_quality({"answer": ""}, {"answer": "some text"}) == 0.0

    def test_empty_fresh_answer(self):
        assert cache_hit_quality({"answer": "some text"}, {"answer": ""}) == 0.0

    def test_both_empty(self):
        assert cache_hit_quality({"answer": ""}, {"answer": ""}) == 0.0

    def test_missing_answer_key(self):
        assert cache_hit_quality({}, {"answer": "text"}) == 0.0

    def test_case_insensitive(self):
        cached = {"answer": "Hello World"}
        fresh = {"answer": "hello world"}
        assert cache_hit_quality(cached, fresh) == 1.0
