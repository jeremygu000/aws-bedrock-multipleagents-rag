"""Unit tests for HyDE retriever."""

import pytest
from unittest.mock import Mock, MagicMock

from app.hyde_retriever import HyDEConfig, HyDERetriever


class TestHyDEConfig:
    """Test HyDE configuration."""

    def test_default_config(self):
        config = HyDEConfig()
        assert config.enabled is True
        assert config.min_query_length == 5
        assert config.temperature == 0.7
        assert config.include_original is True

    def test_custom_config(self):
        config = HyDEConfig(enabled=False, min_query_length=10)
        assert config.enabled is False
        assert config.min_query_length == 10


class TestHyDERetriever:
    """Test HyDE retriever logic."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client."""
        mock = Mock()
        mock.invoke.return_value = Mock(content="This is a hypothetical document about the topic.")
        return mock

    @pytest.fixture
    def mock_embeddings(self):
        """Mock embedding model."""
        mock = Mock()
        mock.embed_query.return_value = [0.1, 0.2, 0.3]  # Dummy embedding
        return mock

    @pytest.fixture
    def hyde_retriever(self, mock_llm, mock_embeddings):
        """Initialize HyDE retriever."""
        config = HyDEConfig(enabled=True, include_original=True)
        return HyDERetriever(mock_llm, mock_embeddings, config)

    def test_short_query_skipped(self, hyde_retriever):
        """Test that short queries (<5 tokens) are skipped."""
        query = "short"
        assert not hyde_retriever._should_use_hyde(query)

    def test_long_query_enabled(self, hyde_retriever):
        """Test that long queries enable HyDE."""
        query = "This is a long query about machine learning and retrieval systems"
        assert hyde_retriever._should_use_hyde(query)

    def test_entity_query_skipped(self, hyde_retriever):
        """Test that entity queries skip HyDE."""
        query = "Tell me about John Smith born in 2000"
        assert not hyde_retriever._should_use_hyde(query)

    def test_reasoning_query_enabled(self, hyde_retriever):
        """Test that reasoning queries enable HyDE."""
        query = "Explain why machine learning works better than traditional methods"
        assert hyde_retriever._should_use_hyde(query)

    def test_generate_hypothesis(self, hyde_retriever, mock_llm):
        """Test hypothesis generation."""
        query = "What is artificial intelligence?"
        hypothesis = hyde_retriever.generate_hypothesis(query)
        assert hypothesis == "This is a hypothetical document about the topic."
        mock_llm.invoke.assert_called_once()

    def test_get_query_embeddings_single_hypothesis(self, hyde_retriever, mock_embeddings):
        """Test single hypothesis embedding generation."""
        query = "Explain machine learning concepts and applications"
        result = hyde_retriever.get_query_embeddings(query)

        assert "embeddings" in result
        assert "sources" in result
        assert "strategy" in result
        # With include_original=True, should have 2 embeddings (hypothesis + original)
        assert len(result["embeddings"]) == 2
        assert "original_query" in result["sources"]

    def test_skip_hyde_returns_original(self, hyde_retriever, mock_embeddings):
        """Test that skipped queries return original embedding only."""
        hyde_retriever.config.enabled = False
        query = "short query"
        result = hyde_retriever.get_query_embeddings(query)

        assert len(result["embeddings"]) == 1
        assert result["sources"] == ["original_query"]
        assert result["strategy"] == "original"

    def test_fallback_on_hypothesis_failure(self, hyde_retriever, mock_llm):
        """Test fallback to original query on hypothesis generation failure."""
        mock_llm.invoke.side_effect = Exception("LLM error")
        query = "Explain complex topics in detail"
        result = hyde_retriever.get_query_embeddings(query)

        # Should still return embeddings (original)
        assert len(result["embeddings"]) > 0


class TestHyDERoutingLogic:
    """Test HyDE routing decisions."""

    def test_entity_detection(self):
        from app.hyde_retriever import HyDERetriever

        mock_llm = Mock()
        mock_embeddings = Mock()
        hyde = HyDERetriever(mock_llm, mock_embeddings)

        # Should detect entities
        assert hyde._has_entities("John Smith works at Google")
        assert hyde._has_entities("Event on 2024-03-15")
        assert hyde._has_entities("ID: WINFxxxx12345")

        # Should not detect entities
        assert not hyde._has_entities("what is machine learning")
        assert not hyde._has_entities("how to build a system")

    def test_reasoning_detection(self):
        from app.hyde_retriever import HyDERetriever

        mock_llm = Mock()
        mock_embeddings = Mock()
        hyde = HyDERetriever(mock_llm, mock_embeddings)

        # Should detect reasoning
        assert hyde._is_reasoning_query("Explain why this works")
        assert hyde._is_reasoning_query("How does the system compare to alternatives?")
        assert hyde._is_reasoning_query("Analyze the impact of this change")

        # Should not detect reasoning
        assert not hyde._is_reasoning_query("What is the date?")
        assert not hyde._is_reasoning_query("Find a document")
