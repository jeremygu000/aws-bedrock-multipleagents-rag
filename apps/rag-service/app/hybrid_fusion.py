"""Weighted reciprocal-rank fusion for graph + traditional retrieval hits (Phase 3.3).

Merges graph-derived chunk hits with traditional retrieval hits using
weighted RRF scoring.  Each source contributes a reciprocal-rank score
scaled by its weight, and hits appearing in both sources are deduplicated
with their scores summed.

Graph hits carry ``entity_context`` / ``relation_context`` metadata so
downstream nodes (reranker, answer generator) can leverage graph evidence.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import GraphContext

logger = logging.getLogger(__name__)


def fuse_graph_and_traditional(
    traditional_hits: list[dict[str, Any]],
    graph_chunk_hits: list[dict[str, Any]],
    graph_context: GraphContext,
    graph_weight: float = 0.6,
    rrf_k: int = 60,
    k_final: int = 8,
) -> list[dict[str, Any]]:
    """Merge traditional retrieval hits with graph-derived chunk hits via weighted RRF.

    Scoring formula per hit:
        ``score = trad_weight / (rrf_k + trad_rank) + graph_weight / (rrf_k + graph_rank)``

    where ``trad_weight = 1.0 - graph_weight``.

    Hits appearing in only one source receive the weighted RRF contribution
    from that source alone.  Duplicate ``chunk_id`` entries are merged —
    the payload from the first occurrence (traditional preferred) is kept.

    Graph-sourced hits are enriched with ``entity_context`` and
    ``relation_context`` fields derived from ``GraphContext``.
    """

    trad_weight = 1.0 - graph_weight

    # Build rank maps (1-indexed)
    trad_rank: dict[str, int] = {
        hit["chunk_id"]: rank for rank, hit in enumerate(traditional_hits, start=1)
    }
    graph_rank: dict[str, int] = {
        hit["chunk_id"]: rank for rank, hit in enumerate(graph_chunk_hits, start=1)
    }

    # Build entity/relation context lookup keyed by source_chunk_id
    entity_context_by_chunk = _build_entity_context_map(graph_context)
    relation_context_by_chunk = _build_relation_context_map(graph_context)

    # Merge payloads — traditional source takes precedence for duplicate chunk_ids
    payload_by_chunk: dict[str, dict[str, Any]] = {}
    for hit in traditional_hits:
        cid = hit["chunk_id"]
        if cid not in payload_by_chunk:
            payload_by_chunk[cid] = _copy_hit(hit)
    for hit in graph_chunk_hits:
        cid = hit["chunk_id"]
        if cid not in payload_by_chunk:
            payload_by_chunk[cid] = _copy_hit(hit)

    # Score and enrich
    scored: list[dict[str, Any]] = []
    for chunk_id, payload in payload_by_chunk.items():
        score = 0.0
        sources: list[str] = []

        trad_pos = trad_rank.get(chunk_id)
        if trad_pos is not None:
            score += trad_weight / (rrf_k + trad_pos)
            sources.append("traditional")

        graph_pos = graph_rank.get(chunk_id)
        if graph_pos is not None:
            score += graph_weight / (rrf_k + graph_pos)
            sources.append("graph")

        payload["score"] = score
        payload["fused_score"] = score
        payload["source"] = "+".join(sources)

        # Attach graph evidence when this chunk is referenced by graph entities/relations
        if chunk_id in entity_context_by_chunk:
            payload["entity_context"] = entity_context_by_chunk[chunk_id]
        if chunk_id in relation_context_by_chunk:
            payload["relation_context"] = relation_context_by_chunk[chunk_id]

        scored.append(payload)

    scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return scored[:k_final]


def _build_entity_context_map(graph_context: GraphContext) -> dict[str, list[dict[str, Any]]]:
    """Map source_chunk_ids to the entities that reference them."""

    result: dict[str, list[dict[str, Any]]] = {}
    for entity in graph_context.entities:
        entity_dict = {"name": entity.name, "type": entity.type, "description": entity.description}
        for cid in graph_context.source_chunk_ids:
            result.setdefault(cid, []).append(entity_dict)
    return result


def _build_relation_context_map(graph_context: GraphContext) -> dict[str, list[dict[str, Any]]]:
    """Map source_chunk_ids to the relations that reference them."""

    result: dict[str, list[dict[str, Any]]] = {}
    for relation in graph_context.relations:
        rel_dict = {
            "source_entity": relation.source_entity,
            "target_entity": relation.target_entity,
            "relation_type": relation.relation_type,
            "evidence": relation.evidence,
        }
        for cid in graph_context.source_chunk_ids:
            result.setdefault(cid, []).append(rel_dict)
    return result


def _copy_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Create a safe mutable copy of a retrieval hit payload."""

    citation = hit.get("citation")
    metadata = hit.get("metadata")
    return {
        **hit,
        "citation": dict(citation) if isinstance(citation, dict) else {},
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
    }
