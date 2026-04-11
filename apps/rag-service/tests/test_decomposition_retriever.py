"""Unit tests for DecompositionRetriever — parallel/sequential retrieval and RRF merge."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.decomposition_retriever import DecompositionRetriever
from app.models import RetrieveRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(chunk_id: str, score: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "chunk_text": f"text for {chunk_id}", "score": score}


def _make_hit_with_id(id_val: str, score: float = 0.5) -> dict:
    return {"id": id_val, "chunk_text": f"text for {id_val}", "score": score}


def _base_request() -> RetrieveRequest:
    return RetrieveRequest(
        query="original query",
        k_sparse=20,
        k_dense=20,
        k_final=5,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.retrieve.return_value = [
        _make_hit("chunk_a"),
        _make_hit("chunk_b"),
    ]
    return repo


@pytest.fixture
def retriever(mock_repository):
    return DecompositionRetriever(mock_repository)


# ---------------------------------------------------------------------------
# Parallel retrieval
# ---------------------------------------------------------------------------


class TestParallelRetrieval:

    def test_empty_sub_questions_returns_empty(self, retriever):
        result = asyncio.run(
            retriever.retrieve_sub_questions_parallel([], _base_request())
        )
        assert result == []

    def test_single_sub_question_returns_one_result_list(self, retriever, mock_repository):
        mock_repository.retrieve.return_value = [_make_hit("c1")]
        result = asyncio.run(
            retriever.retrieve_sub_questions_parallel(["sub q1"], _base_request())
        )
        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["chunk_id"] == "c1"

    def test_multiple_sub_questions_all_succeed(self, retriever, mock_repository):
        mock_repository.retrieve.side_effect = [
            [_make_hit("a1"), _make_hit("a2")],
            [_make_hit("b1")],
            [_make_hit("c1"), _make_hit("c2"), _make_hit("c3")],
        ]
        result = asyncio.run(
            retriever.retrieve_sub_questions_parallel(
                ["q1", "q2", "q3"], _base_request()
            )
        )
        assert len(result) == 3
        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert len(result[2]) == 3

    def test_partial_failure_returns_empty_for_failed(self, retriever, mock_repository):
        mock_repository.retrieve.side_effect = [
            [_make_hit("ok1")],
            RuntimeError("DB connection lost"),
            [_make_hit("ok2")],
        ]
        result = asyncio.run(
            retriever.retrieve_sub_questions_parallel(
                ["q1", "q2", "q3"], _base_request()
            )
        )
        assert len(result) == 3
        assert len(result[0]) == 1  # succeeded
        assert result[1] == []  # failed → empty
        assert len(result[2]) == 1  # succeeded

    def test_all_failures_returns_all_empty(self, retriever, mock_repository):
        mock_repository.retrieve.side_effect = RuntimeError("total failure")
        result = asyncio.run(
            retriever.retrieve_sub_questions_parallel(
                ["q1", "q2"], _base_request()
            )
        )
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

    def test_clones_request_with_sub_question(self, retriever, mock_repository):
        mock_repository.retrieve.return_value = []
        asyncio.run(
            retriever.retrieve_sub_questions_parallel(
                ["sub q1", "sub q2"], _base_request()
            )
        )
        assert mock_repository.retrieve.call_count == 2
        first_call_req = mock_repository.retrieve.call_args_list[0][0][0]
        second_call_req = mock_repository.retrieve.call_args_list[1][0][0]
        assert first_call_req.query == "sub q1"
        assert second_call_req.query == "sub q2"

    def test_cloned_request_preserves_base_fields(self, retriever, mock_repository):
        base = _base_request()
        mock_repository.retrieve.return_value = []
        asyncio.run(
            retriever.retrieve_sub_questions_parallel(["sq"], base)
        )
        cloned = mock_repository.retrieve.call_args_list[0][0][0]
        assert cloned.k_sparse == base.k_sparse
        assert cloned.k_dense == base.k_dense
        assert cloned.k_final == base.k_final


# ---------------------------------------------------------------------------
# Sequential retrieval
# ---------------------------------------------------------------------------


class TestSequentialRetrieval:

    def test_empty_sub_questions_returns_empty(self, retriever):
        result = retriever.retrieve_sub_questions_sequential([], _base_request())
        assert result == []

    def test_all_succeed(self, retriever, mock_repository):
        mock_repository.retrieve.side_effect = [
            [_make_hit("a1")],
            [_make_hit("b1"), _make_hit("b2")],
        ]
        result = retriever.retrieve_sub_questions_sequential(
            ["q1", "q2"], _base_request()
        )
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 2

    def test_partial_failure_returns_empty_for_failed(self, retriever, mock_repository):
        mock_repository.retrieve.side_effect = [
            [_make_hit("ok")],
            RuntimeError("fail"),
        ]
        result = retriever.retrieve_sub_questions_sequential(
            ["q1", "q2"], _base_request()
        )
        assert len(result) == 2
        assert len(result[0]) == 1
        assert result[1] == []

    def test_clones_request_with_sub_question(self, retriever, mock_repository):
        mock_repository.retrieve.return_value = []
        retriever.retrieve_sub_questions_sequential(
            ["sq1", "sq2"], _base_request()
        )
        assert mock_repository.retrieve.call_count == 2
        first_req = mock_repository.retrieve.call_args_list[0][0][0]
        assert first_req.query == "sq1"


# ---------------------------------------------------------------------------
# Merge sub-question hits (RRF)
# ---------------------------------------------------------------------------


class TestMergeHits:

    def test_empty_inputs_returns_empty(self):
        result = DecompositionRetriever.merge_sub_question_hits([], [], k_final=5)
        assert result == []

    def test_only_original_hits(self):
        original = [_make_hit("a"), _make_hit("b")]
        result = DecompositionRetriever.merge_sub_question_hits(
            original, [], k_final=5
        )
        assert len(result) == 2
        assert result[0]["chunk_id"] == "a"  # rank 1 → higher RRF score
        assert result[1]["chunk_id"] == "b"

    def test_only_sub_question_hits(self):
        sub_hits = [[_make_hit("s1"), _make_hit("s2")]]
        result = DecompositionRetriever.merge_sub_question_hits(
            [], sub_hits, k_final=5
        )
        assert len(result) == 2
        assert result[0]["chunk_id"] == "s1"

    def test_dedup_original_takes_priority(self):
        """When same chunk_id appears in both, original hit is kept."""
        original = [{"chunk_id": "dup", "chunk_text": "original version", "score": 0.9}]
        sub_hits = [[{"chunk_id": "dup", "chunk_text": "sub version", "score": 0.1}]]
        result = DecompositionRetriever.merge_sub_question_hits(
            original, sub_hits, k_final=5
        )
        assert len(result) == 1
        assert result[0]["chunk_text"] == "original version"

    def test_rrf_scores_accumulate_for_shared_chunks(self):
        """Chunk appearing in original + sub-question gets combined score → higher rank."""
        original = [_make_hit("shared"), _make_hit("only_orig")]
        sub_hits = [[_make_hit("shared"), _make_hit("only_sub")]]
        result = DecompositionRetriever.merge_sub_question_hits(
            original, sub_hits, k_final=10
        )
        # "shared" should be first because it has accumulated RRF from both sources
        assert result[0]["chunk_id"] == "shared"

    def test_k_final_truncation(self):
        original = [_make_hit(f"c{i}") for i in range(20)]
        result = DecompositionRetriever.merge_sub_question_hits(
            original, [], k_final=5
        )
        assert len(result) == 5

    def test_original_weight_affects_ranking(self):
        original = [_make_hit("orig1")]
        sub_hits = [[_make_hit("sub1")]]
        result_high_orig = DecompositionRetriever.merge_sub_question_hits(
            original, sub_hits, original_weight=10.0, sub_question_weight=0.1, k_final=2
        )
        assert result_high_orig[0]["chunk_id"] == "orig1"

    def test_sub_question_weight_affects_ranking(self):
        original = [_make_hit("orig1")]
        sub_hits = [[_make_hit("sub1")]]
        result_high_sub = DecompositionRetriever.merge_sub_question_hits(
            original, sub_hits, original_weight=0.1, sub_question_weight=10.0, k_final=2
        )
        assert result_high_sub[0]["chunk_id"] == "sub1"

    def test_multiple_sub_question_lists(self):
        original = [_make_hit("orig")]
        sub_hits = [
            [_make_hit("sq1_a"), _make_hit("common")],
            [_make_hit("sq2_a"), _make_hit("common")],
        ]
        result = DecompositionRetriever.merge_sub_question_hits(
            original, sub_hits, k_final=10
        )
        chunk_ids = [h["chunk_id"] for h in result]
        assert "common" in chunk_ids
        assert "orig" in chunk_ids
        assert "sq1_a" in chunk_ids
        assert "sq2_a" in chunk_ids

    def test_hit_with_id_field_instead_of_chunk_id(self):
        original = [_make_hit_with_id("id_only")]
        result = DecompositionRetriever.merge_sub_question_hits(
            original, [], k_final=5
        )
        assert len(result) == 1
        assert result[0]["id"] == "id_only"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_parallel_with_timeout_param(self, retriever, mock_repository):
        mock_repository.retrieve.return_value = [_make_hit("t1")]
        result = asyncio.run(
            retriever.retrieve_sub_questions_parallel(
                ["q"], _base_request(), timeout_seconds=5
            )
        )
        assert len(result) == 1

    def test_merge_preserves_extra_hit_fields(self):
        hit = {"chunk_id": "x", "chunk_text": "text", "score": 0.5, "custom_field": 42}
        result = DecompositionRetriever.merge_sub_question_hits(
            [hit], [], k_final=5
        )
        assert result[0]["custom_field"] == 42

    def test_merge_with_hits_missing_chunk_id_and_id(self):
        """Hits with no 'chunk_id' and no 'id' get auto-generated IDs for scoring
        but are excluded from hit_map (since chunk_id is None)."""
        hit_no_id = {"chunk_text": "orphan", "score": 0.5}
        result = DecompositionRetriever.merge_sub_question_hits(
            [hit_no_id], [], k_final=5
        )
        # Auto-generated fallback ID "chunk_1" is used in rrf_scores
        # but hit_map only stores if chunk_id is truthy — hit_no_id has no chunk_id/id
        # so it won't be in hit_map and won't appear in merged results
        assert result == []

    def test_sequential_preserves_order(self, retriever, mock_repository):
        mock_repository.retrieve.side_effect = [
            [_make_hit("first")],
            [_make_hit("second")],
            [_make_hit("third")],
        ]
        result = retriever.retrieve_sub_questions_sequential(
            ["q1", "q2", "q3"], _base_request()
        )
        assert result[0][0]["chunk_id"] == "first"
        assert result[1][0]["chunk_id"] == "second"
        assert result[2][0]["chunk_id"] == "third"
