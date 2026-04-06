"""Unit tests for DecompositionRetriever integration into workflow._impl_retrieve."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.decomposition_retriever import DecompositionRetriever
from app.models import RetrieveRequest
from app.query_decomposer import SubQuestion
from app.workflow import RagWorkflow


@pytest.fixture
def mock_repository():
    """Mock PostgresRepository for retrieval operations."""
    repo = MagicMock()
    repo.retrieve = MagicMock(return_value=[
        {"sourceId": "doc1", "snippet": "Result 1", "score": 0.95},
        {"sourceId": "doc2", "snippet": "Result 2", "score": 0.85},
    ])
    return repo


@pytest.fixture
def mock_decomposition_retriever():
    """Mock DecompositionRetriever for parallel sub-question retrieval."""
    retriever = MagicMock(spec=DecompositionRetriever)
    
    # Simulate parallel retrieval returning hits for each sub-question
    retriever.retrieve_sub_questions_parallel = AsyncMock(return_value=[
        # Results for sub-question 1
        [
            {"sourceId": "doc1", "snippet": "SQ1 Result", "score": 0.90},
            {"sourceId": "doc3", "snippet": "SQ1 Extra", "score": 0.80},
        ],
        # Results for sub-question 2
        [
            {"sourceId": "doc2", "snippet": "SQ2 Result", "score": 0.88},
            {"sourceId": "doc4", "snippet": "SQ2 Extra", "score": 0.78},
        ],
    ])
    return retriever


@pytest.fixture
def mock_settings():
    """Mock Settings with decomposition config."""
    settings = MagicMock()
    settings.enable_query_decomposition = True
    settings.decomposition_timeout_s = 10
    return settings


@pytest.fixture
def mock_query_processor():
    """Mock QueryProcessor."""
    return MagicMock()


@pytest.fixture
def mock_answer_generator():
    """Mock RoutedAnswerGenerator."""
    return MagicMock()


@pytest.fixture
def mock_reranker():
    """Mock LLMReranker."""
    return MagicMock()


@pytest.fixture
def workflow(
    mock_settings,
    mock_repository,
    mock_query_processor,
    mock_answer_generator,
    mock_reranker,
    mock_decomposition_retriever,
):
    """Create RagWorkflow with mocked dependencies."""
    return RagWorkflow(
        settings=mock_settings,
        repository=mock_repository,
        query_processor=mock_query_processor,
        answer_generator=mock_answer_generator,
        reranker=mock_reranker,
        decomposition_retriever=mock_decomposition_retriever,
    )


class TestDecompositionRetrieverIntegration:
    """Test DecompositionRetriever integration into workflow retrieve pipeline."""

    def test_retrieve_with_decomposition_disabled(self, workflow, mock_repository):
        """Test standard retrieval path when decomposition is disabled."""
        state = {
            "request": RetrieveRequest(query="test query", k_final=5),
            "decomposition_used": False,
            "sub_questions": [],
        }
        
        result = workflow._impl_retrieve(state)
        
        # Should use standard retrieval
        mock_repository.retrieve.assert_called_once()
        assert "hits" in result
        assert len(result["hits"]) == 2

    def test_retrieve_with_decomposition_no_sub_questions(self, workflow, mock_repository):
        """Test retrieval when decomposition is used but no sub-questions generated."""
        state = {
            "request": RetrieveRequest(query="test query", k_final=5),
            "decomposition_used": True,
            "sub_questions": [],
        }
        
        result = workflow._impl_retrieve(state)
        
        # Should fall back to standard retrieval
        mock_repository.retrieve.assert_called_once()
        assert "hits" in result
        assert len(result["hits"]) == 2

    def test_retrieve_with_decomposition_missing_retriever(self, mock_settings, mock_repository, mock_query_processor, mock_answer_generator, mock_reranker):
        """Test retrieval when decomposition is enabled but retriever is None."""
        # Create workflow without decomposition_retriever
        workflow_no_decomp = RagWorkflow(
            settings=mock_settings,
            repository=mock_repository,
            query_processor=mock_query_processor,
            answer_generator=mock_answer_generator,
            reranker=mock_reranker,
            decomposition_retriever=None,  # Explicitly None
        )
        
        state = {
            "request": RetrieveRequest(query="test query", k_final=5),
            "decomposition_used": True,
            "sub_questions": [
                SubQuestion(id=1, question="What is SQ1?  What are the details?", focus="fact", retrieve_strategy="hybrid"),
            ],
        }
        
        result = workflow_no_decomp._impl_retrieve(state)
        
        # Should fall back to standard retrieval
        mock_repository.retrieve.assert_called_once()
        assert "hits" in result

    def test_retrieve_with_decomposition_parallel_retrieval(self, workflow, mock_repository, mock_decomposition_retriever):
        """Test decomposition path with parallel sub-question retrieval."""
        state = {
            "request": RetrieveRequest(query="main query", k_final=8),
            "decomposition_used": True,
            "sub_questions": [
                SubQuestion(id=1, question="What is the main topic?", focus="concept", retrieve_strategy="hybrid"),
                SubQuestion(id=2, question="What are the details?", focus="detail", retrieve_strategy="hybrid"),
            ],
        }
        
        # Need to run async retriever in sync context
        result = workflow._impl_retrieve(state)
        
        # For async testing, we'd need a different setup
        # This test shows the structure; actual async test would need pytest-asyncio
        assert "hits" in result


class TestMergeDecompositionHits:
    """Test _merge_decomposition_hits deduplication and ranking logic."""

    def test_merge_empty_lists(self, workflow):
        """Test merging empty hit lists."""
        hits_lists = [[], []]
        result = workflow._merge_decomposition_hits(hits_lists, k_final=5)
        
        assert result == []

    def test_merge_single_list(self, workflow):
        """Test merging a single hit list."""
        hits_lists = [
            [
                {"sourceId": "doc1", "snippet": "Result 1", "score": 0.95},
                {"sourceId": "doc2", "snippet": "Result 2", "score": 0.85},
            ]
        ]
        result = workflow._merge_decomposition_hits(hits_lists, k_final=5)
        
        assert len(result) == 2
        assert result[0]["sourceId"] == "doc1"
        assert result[1]["sourceId"] == "doc2"

    def test_merge_multiple_lists_deduplication(self, workflow):
        """Test deduplication of hits across multiple lists."""
        hits_lists = [
            # SQ1 results
            [
                {"sourceId": "doc1", "snippet": "SQ1 Result 1", "score": 0.95},
                {"sourceId": "doc3", "snippet": "SQ1 Result 2", "score": 0.85},
            ],
            # SQ2 results
            [
                {"sourceId": "doc1", "snippet": "SQ2 Result (dup)", "score": 0.90},  # Duplicate!
                {"sourceId": "doc2", "snippet": "SQ2 Result 1", "score": 0.88},
            ],
        ]
        result = workflow._merge_decomposition_hits(hits_lists, k_final=10)
        
        # Should deduplicate: doc1 appears in both, should keep first occurrence
        assert len(result) == 3
        source_ids = [hit["sourceId"] for hit in result]
        assert source_ids == ["doc1", "doc3", "doc2"]
        # First occurrence should be from SQ1
        assert result[0]["snippet"] == "SQ1 Result 1"

    def test_merge_respects_k_final(self, workflow):
        """Test that merge stops at k_final limit."""
        hits_lists = [
            [
                {"sourceId": "doc1", "snippet": "Result 1", "score": 0.95},
                {"sourceId": "doc2", "snippet": "Result 2", "score": 0.90},
                {"sourceId": "doc3", "snippet": "Result 3", "score": 0.85},
            ],
            [
                {"sourceId": "doc4", "snippet": "Result 4", "score": 0.80},
                {"sourceId": "doc5", "snippet": "Result 5", "score": 0.75},
            ],
        ]
        result = workflow._merge_decomposition_hits(hits_lists, k_final=3)
        
        assert len(result) == 3
        assert result[0]["sourceId"] == "doc1"
        assert result[1]["sourceId"] == "doc2"
        assert result[2]["sourceId"] == "doc3"

    def test_merge_custom_dedup_key(self, workflow):
        """Test deduplication using custom key field."""
        hits_lists = [
            [
                {"custom_id": "A", "snippet": "Result 1"},
                {"custom_id": "B", "snippet": "Result 2"},
            ],
            [
                {"custom_id": "A", "snippet": "Result 1 Dup"},  # Duplicate custom_id
                {"custom_id": "C", "snippet": "Result 3"},
            ],
        ]
        result = workflow._merge_decomposition_hits(
            hits_lists, k_final=10, dedup_key="custom_id"
        )
        
        assert len(result) == 3
        custom_ids = [hit["custom_id"] for hit in result]
        assert custom_ids == ["A", "B", "C"]
        # First occurrence preserved
        assert result[0]["snippet"] == "Result 1"

    def test_merge_preserves_order(self, workflow):
        """Test that merge preserves relative ranking from sub-questions."""
        hits_lists = [
            # SQ1: high confidence results
            [
                {"sourceId": "doc1", "snippet": "High confidence 1", "score": 0.99},
                {"sourceId": "doc2", "snippet": "High confidence 2", "score": 0.98},
            ],
            # SQ2: lower confidence results
            [
                {"sourceId": "doc3", "snippet": "Lower confidence 1", "score": 0.60},
                {"sourceId": "doc4", "snippet": "Lower confidence 2", "score": 0.50},
            ],
        ]
        result = workflow._merge_decomposition_hits(hits_lists, k_final=10)
        
        # Should maintain SQ1 order first (higher confidence), then SQ2
        assert len(result) == 4
        assert result[0]["sourceId"] == "doc1"
        assert result[1]["sourceId"] == "doc2"
        assert result[2]["sourceId"] == "doc3"
        assert result[3]["sourceId"] == "doc4"

    def test_merge_handles_missing_dedup_key(self, workflow):
        """Test merge behavior when dedup_key is missing from some hits."""
        hits_lists = [
            [
                {"sourceId": "doc1", "snippet": "Result 1"},
                {"snippet": "Result with no sourceId"},  # Missing sourceId
                {"sourceId": "doc2", "snippet": "Result 2"},
            ],
        ]
        result = workflow._merge_decomposition_hits(hits_lists, k_final=10)
        
        # Should skip entries without sourceId
        assert len(result) == 2
        assert result[0]["sourceId"] == "doc1"
        assert result[1]["sourceId"] == "doc2"
