from __future__ import annotations

from app.hybrid_fusion import fuse_graph_and_traditional
from app.models import GraphContext, GraphEntity, GraphRelation


def _make_hit(chunk_id: str, score: float = 0.1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc1",
        "chunk_text": f"text for {chunk_id}",
        "score": score,
        "category": "test",
        "lang": "en",
        "source_type": "crawler",
        "metadata": {},
        "citation": {
            "url": "https://example.com",
            "title": "Test",
            "year": 2025,
            "month": 3,
        },
    }


def _make_graph_context(
    chunk_ids: list[str] | None = None,
    entities: list[GraphEntity] | None = None,
    relations: list[GraphRelation] | None = None,
) -> GraphContext:
    return GraphContext(
        entities=entities or [],
        relations=relations or [],
        source_chunk_ids=chunk_ids or [],
    )


# ---------------------------------------------------------------------------
# Basic RRF scoring
# ---------------------------------------------------------------------------


def test_fuse_traditional_only_when_no_graph_hits() -> None:
    trad = [_make_hit("c1"), _make_hit("c2")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=[],
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c1"
    assert result[0]["source"] == "traditional"


def test_fuse_graph_only_when_no_traditional_hits() -> None:
    graph_hits = [_make_hit("g1"), _make_hit("g2")]
    result = fuse_graph_and_traditional(
        traditional_hits=[],
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(chunk_ids=["g1", "g2"]),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert len(result) == 2
    assert result[0]["chunk_id"] == "g1"
    assert result[0]["source"] == "graph"


def test_fuse_both_sources_scored_with_weights() -> None:
    trad = [_make_hit("c1")]
    graph_hits = [_make_hit("g1")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert len(result) == 2
    trad_score = 0.4 / (60 + 1)
    graph_score = 0.6 / (60 + 1)
    assert (
        abs(result[0]["score"] - graph_score) < 1e-9 or abs(result[0]["score"] - trad_score) < 1e-9
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_fuse_dedup_same_chunk_id_merges_scores() -> None:
    trad = [_make_hit("shared")]
    graph_hits = [_make_hit("shared")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert len(result) == 1
    expected_score = 0.4 / (60 + 1) + 0.6 / (60 + 1)
    assert abs(result[0]["score"] - expected_score) < 1e-9
    assert result[0]["source"] == "traditional+graph"


def test_fuse_dedup_traditional_payload_takes_precedence() -> None:
    trad = [_make_hit("shared")]
    trad[0]["chunk_text"] = "traditional text"
    graph_hits = [_make_hit("shared")]
    graph_hits[0]["chunk_text"] = "graph text"
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert result[0]["chunk_text"] == "traditional text"


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------


def test_fuse_graph_weight_affects_ranking() -> None:
    trad = [_make_hit("c1")]
    graph_hits = [_make_hit("g1")]

    result_high_graph = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.9,
        rrf_k=60,
        k_final=5,
    )
    result_low_graph = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.1,
        rrf_k=60,
        k_final=5,
    )
    assert result_high_graph[0]["chunk_id"] == "g1"
    assert result_low_graph[0]["chunk_id"] == "c1"


def test_fuse_equal_weight_same_rank_produces_equal_scores() -> None:
    trad = [_make_hit("c1")]
    graph_hits = [_make_hit("g1")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.5,
        rrf_k=60,
        k_final=5,
    )
    assert abs(result[0]["score"] - result[1]["score"]) < 1e-9


# ---------------------------------------------------------------------------
# k_final truncation
# ---------------------------------------------------------------------------


def test_fuse_respects_k_final() -> None:
    trad = [_make_hit(f"c{i}") for i in range(10)]
    graph_hits = [_make_hit(f"g{i}") for i in range(10)]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Entity/relation context enrichment
# ---------------------------------------------------------------------------


def test_fuse_attaches_entity_context_to_graph_chunks() -> None:
    entities = [GraphEntity(entity_id="e1", name="Foo", type="Work", description="A work")]
    ctx = _make_graph_context(
        chunk_ids=["g1"],
        entities=entities,
    )
    graph_hits = [_make_hit("g1")]
    result = fuse_graph_and_traditional(
        traditional_hits=[],
        graph_chunk_hits=graph_hits,
        graph_context=ctx,
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert "entity_context" in result[0]
    assert result[0]["entity_context"][0]["name"] == "Foo"


def test_fuse_attaches_relation_context_to_graph_chunks() -> None:
    relations = [
        GraphRelation(
            source_entity="A",
            target_entity="B",
            relation_type="WROTE",
            evidence="A wrote B",
        )
    ]
    ctx = _make_graph_context(
        chunk_ids=["g1"],
        relations=relations,
    )
    graph_hits = [_make_hit("g1")]
    result = fuse_graph_and_traditional(
        traditional_hits=[],
        graph_chunk_hits=graph_hits,
        graph_context=ctx,
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert "relation_context" in result[0]
    assert result[0]["relation_context"][0]["relation_type"] == "WROTE"


def test_fuse_no_entity_context_for_traditional_only_chunks() -> None:
    entities = [GraphEntity(entity_id="e1", name="Foo", type="Work", description="desc")]
    ctx = _make_graph_context(chunk_ids=["g1"], entities=entities)
    trad = [_make_hit("c1")]
    graph_hits = [_make_hit("g1")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=ctx,
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    trad_hit = next(h for h in result if h["chunk_id"] == "c1")
    assert "entity_context" not in trad_hit


# ---------------------------------------------------------------------------
# fused_score and source provenance
# ---------------------------------------------------------------------------


def test_fuse_sets_fused_score_field() -> None:
    trad = [_make_hit("c1")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=[],
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert "fused_score" in result[0]
    assert result[0]["fused_score"] == result[0]["score"]


def test_fuse_source_provenance_values() -> None:
    trad = [_make_hit("c1"), _make_hit("shared")]
    graph_hits = [_make_hit("shared"), _make_hit("g1")]
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=graph_hits,
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    sources = {h["chunk_id"]: h["source"] for h in result}
    assert sources["c1"] == "traditional"
    assert sources["g1"] == "graph"
    assert sources["shared"] == "traditional+graph"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_fuse_empty_both_sources() -> None:
    result = fuse_graph_and_traditional(
        traditional_hits=[],
        graph_chunk_hits=[],
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert result == []


def test_fuse_preserves_citation_and_metadata() -> None:
    trad = [_make_hit("c1")]
    trad[0]["citation"]["title"] = "Important Doc"
    trad[0]["metadata"]["custom_field"] = "value"
    result = fuse_graph_and_traditional(
        traditional_hits=trad,
        graph_chunk_hits=[],
        graph_context=_make_graph_context(),
        graph_weight=0.6,
        rrf_k=60,
        k_final=5,
    )
    assert result[0]["citation"]["title"] == "Important Doc"
    assert result[0]["metadata"]["custom_field"] == "value"
