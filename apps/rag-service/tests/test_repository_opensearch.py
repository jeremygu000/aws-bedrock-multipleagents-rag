from __future__ import annotations

from app.config import Settings
from app.models import RetrieveRequest
from app.repository import (
    PostgresRepository,
    _build_opensearch_filters,
    _normalize_opensearch_hit,
)


def _sample_hit(chunk_id: str = "c1") -> dict:
    return {
        "_id": chunk_id,
        "_source": {
            "chunk_id": chunk_id,
            "doc_id": "d1",
            "chunk_text": "policy snippet",
            "category": "policy",
            "lang": "en",
            "source_type": "crawler",
            "metadata": {"source": "web"},
            "citation": {
                "url": "https://example.com/policy",
                "title": "Policy Doc",
                "year": 2025,
                "month": 3,
                "page_start": 1,
                "page_end": 2,
            },
        },
    }


def _normalized_sample(chunk_id: str) -> dict:
    normalized = _normalize_opensearch_hit(_sample_hit(chunk_id), require_strict_citation=True)
    assert normalized is not None
    return normalized


def test_normalize_opensearch_hit_with_nested_citation() -> None:
    normalized = _normalize_opensearch_hit(_sample_hit(), require_strict_citation=True)
    assert normalized is not None
    assert normalized["chunk_id"] == "c1"
    assert normalized["citation"]["url"] == "https://example.com/policy"
    assert normalized["citation"]["year"] == 2025


def test_normalize_opensearch_hit_strict_mode_rejects_missing_citation() -> None:
    hit = _sample_hit()
    del hit["_source"]["citation"]["url"]
    normalized = _normalize_opensearch_hit(hit, require_strict_citation=True)
    assert normalized is None


def test_build_opensearch_filters_includes_metadata_constraints() -> None:
    request = RetrieveRequest(
        query="policy",
        filters={
            "category": "policy",
            "lang": "en",
            "source_type": "crawler",
            "citation_year_from": 2024,
            "citation_month": 3,
        },
    )
    filters = _build_opensearch_filters(request)
    serialized = str(filters)
    assert "category.keyword" in serialized
    assert "citation.year" in serialized
    assert "citation.month" in serialized


def test_fuse_ranked_hits_uses_rrf_scores() -> None:
    repo = PostgresRepository(
        Settings(
            RAG_SPARSE_BACKEND="opensearch",
            RAG_OPENSEARCH_ENDPOINT="https://example.com",
        )
    )
    sparse_hits = [
        {**_normalized_sample("c1"), "score": 0.0},
        {**_normalized_sample("c2"), "score": 0.0},
    ]
    dense_hits = [{**_normalized_sample("c2"), "score": 0.0}]
    fused = repo._fuse_ranked_hits(
        sparse_hits=sparse_hits, dense_hits=dense_hits, k_final=2, rrf_k=60
    )
    assert len(fused) == 2
    assert fused[0]["chunk_id"] == "c2"
