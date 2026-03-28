"""Custom graph-aware evaluation metrics for RAG pipeline quality measurement.

These metrics are not provided by RAGAS or DeepEval and capture graph-specific
retrieval quality signals unique to our Neo4j + pgvector dual-storage architecture.
"""

from __future__ import annotations

from typing import Any


def graph_traversal_depth(state: dict[str, Any]) -> int:
    """Count max Neo4j hops that contributed to the answer.

    Inspects the ``graph_context`` in workflow state to determine how deep
    the graph traversal went. Returns 0 when graph retrieval was not used.
    """
    graph_ctx = state.get("graph_context")
    if not graph_ctx:
        return 0

    if hasattr(graph_ctx, "entities"):
        has_entities = bool(graph_ctx.entities)
        has_relations = bool(graph_ctx.relations)
    else:
        has_entities = bool(graph_ctx.get("entities"))
        has_relations = bool(graph_ctx.get("relations"))

    if not has_entities and not has_relations:
        return 0

    if has_relations:
        return 2

    return 1


def fusion_contribution_ratio(state: dict[str, Any]) -> dict[str, float]:
    """Calculate fraction of reranked hits from graph vs vector retrieval.

    Returns a dict with ``graph`` and ``vector`` keys, each a float [0.0, 1.0].
    """
    hits = state.get("reranked_hits", [])
    if not hits:
        return {"graph": 0.0, "vector": 0.0}

    graph_count = sum(1 for h in hits if h.get("source") == "graph")
    total = len(hits)
    return {
        "graph": graph_count / total,
        "vector": (total - graph_count) / total,
    }


def multi_source_recall(state: dict[str, Any]) -> int:
    """Count distinct source documents referenced in reranked hits."""
    hits = state.get("reranked_hits", [])
    return len(
        {
            h.get("doc_id") or h.get("document_id")
            for h in hits
            if (h.get("doc_id") or h.get("document_id"))
        }
    )


def cache_hit_quality(
    cached_state: dict[str, Any],
    fresh_state: dict[str, Any],
) -> float:
    """Compare cached answer against fresh answer using simple token overlap.

    Returns a Jaccard similarity score [0.0, 1.0] between the two answers.
    This is a lightweight proxy; full semantic comparison requires an LLM judge.
    """
    cached_answer = (cached_state.get("answer") or "").lower().split()
    fresh_answer = (fresh_state.get("answer") or "").lower().split()

    if not cached_answer or not fresh_answer:
        return 0.0

    cached_set = set(cached_answer)
    fresh_set = set(fresh_answer)

    intersection = cached_set & fresh_set
    union = cached_set | fresh_set

    return len(intersection) / len(union) if union else 0.0
