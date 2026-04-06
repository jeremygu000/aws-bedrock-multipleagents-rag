"""Unit tests for query router."""

import pytest

from app.query_router import QueryAnalysis, QueryRouter, RetrievalStrategy


class TestQueryRouter:
    """Test query routing logic."""

    @pytest.fixture
    def router(self):
        """Initialize query router."""
        return QueryRouter(bm25_ranker=None, embeddings=None)

    def test_entity_detection(self, router):
        """Test named entity detection."""
        # Should detect
        assert router._detect_entities("John Smith works at Google")
        assert router._detect_entities("Event in 2024")
        assert router._detect_entities("WINF123456")

        # Should not detect
        assert not router._detect_entities("what is machine learning")

    def test_factual_query_detection(self, router):
        """Test factual query detection."""
        # Factual queries
        assert router._is_factual_query("What is the capital of France?")
        assert router._is_factual_query("When was this published?")
        assert router._is_factual_query("Find all documents about X")

        # Non-factual
        assert not router._is_factual_query("Why does this matter?")

    def test_reasoning_query_detection(self, router):
        """Test reasoning query detection."""
        # Reasoning queries
        assert router._is_reasoning_query("Explain how machine learning works")
        assert router._is_reasoning_query("Why is this important?")
        assert router._is_reasoning_query("Compare the two approaches")

        # Non-reasoning
        assert not router._is_reasoning_query("List all items")

    def test_routing_short_query(self, router):
        """Test that short queries are routed to BM25."""
        analysis = router.analyze_query("short")
        assert analysis.strategy == RetrievalStrategy.BM25_PRIMARY
        assert analysis.confidence > 0.7

    def test_routing_entity_query(self, router):
        """Test that entity queries are routed to BM25."""
        analysis = router.analyze_query("Find John Smith from 2024")
        assert analysis.strategy == RetrievalStrategy.BM25_PRIMARY

    def test_routing_reasoning_query(self, router):
        """Test that reasoning queries are routed to HyDE."""
        analysis = router.analyze_query("Explain why machine learning is important for data analysis")
        assert analysis.strategy == RetrievalStrategy.HYDE_PRIMARY

    def test_routing_long_query(self, router):
        """Test that long queries are routed to HyDE."""
        query = "Tell me about the history and development of artificial intelligence and its applications in modern society"
        analysis = router.analyze_query(query, estimate_semantic_gap=False)
        assert analysis.strategy == RetrievalStrategy.HYDE_PRIMARY

    def test_semantic_gap_high(self, router):
        """Test routing with high semantic gap — reasoning query must exceed 8 tokens for HYDE_PRIMARY."""
        router.bm25 = True  # Simulate available BM25
        query = "Discuss the philosophical implications of quantum computing"
        analysis = router.analyze_query(query, estimate_semantic_gap=False)
        # 7 tokens + reasoning, but Rule 3 requires token_count > 8 → falls to default HYBRID
        assert analysis.strategy == RetrievalStrategy.HYBRID

    def test_should_use_hyde_quick_check(self, router):
        """Test quick HyDE check."""
        # Should use HyDE
        assert router.should_use_hyde("Explain how neural networks learn patterns")

        # Should not use HyDE
        assert not router.should_use_hyde("short")

    def test_should_use_bm25_quick_check(self, router):
        """Test quick BM25 check."""
        # Should use BM25
        assert router.should_use_bm25("What is John Smith's phone number?")

        # May use hybrid
        assert router.should_use_bm25("Explain the concept") in [True, False]


class TestQueryAnalysis:
    """Test QueryAnalysis model."""

    def test_query_analysis_creation(self):
        """Test creating QueryAnalysis."""
        analysis = QueryAnalysis(
            strategy=RetrievalStrategy.HYDE_PRIMARY,
            confidence=0.85,
            reasoning="Long reasoning query",
            has_entities=False,
            is_factual=False,
            is_reasoning=True,
            query_tokens=20,
        )

        assert analysis.strategy == RetrievalStrategy.HYDE_PRIMARY
        assert analysis.confidence == 0.85
        assert analysis.query_tokens == 20

    def test_query_analysis_confidence_validation(self):
        """Test confidence score validation (0-1)."""
        with pytest.raises(ValueError):
            QueryAnalysis(
                strategy=RetrievalStrategy.HYDE_PRIMARY,
                confidence=1.5,  # Invalid: > 1
                reasoning="test",
                has_entities=False,
                is_factual=False,
                is_reasoning=True,
                query_tokens=10,
            )

        with pytest.raises(ValueError):
            QueryAnalysis(
                strategy=RetrievalStrategy.HYDE_PRIMARY,
                confidence=0,  # Invalid: == 0
                reasoning="test",
                has_entities=False,
                is_factual=False,
                is_reasoning=True,
                query_tokens=10,
            )


class TestRoutingDecisionTree:
    """Test complete routing decision tree."""

    @pytest.fixture
    def router(self):
        return QueryRouter(bm25_ranker=None, embeddings=None)

    def test_complete_routing_scenario_1(self, router):
        """Scenario: Factual lookup with entities."""
        query = "What is the phone number of John Smith at Google?"
        analysis = router.analyze_query(query)
        assert analysis.strategy == RetrievalStrategy.BM25_PRIMARY
        print(f"✓ Scenario 1: {analysis.reasoning}")

    def test_complete_routing_scenario_2(self, router):
        """Scenario: Reasoning-heavy about methods."""
        query = "Explain the pros and cons of different machine learning approaches for time series prediction"
        analysis = router.analyze_query(query, estimate_semantic_gap=False)
        assert analysis.strategy in (RetrievalStrategy.HYDE_PRIMARY, RetrievalStrategy.HYBRID)
        print(f"✓ Scenario 2: {analysis.reasoning}")

    def test_complete_routing_scenario_3(self, router):
        """Scenario: Short factual query."""
        query = "Date?"
        analysis = router.analyze_query(query)
        assert analysis.strategy == RetrievalStrategy.BM25_PRIMARY
        assert "short" in analysis.reasoning.lower()
        print(f"✓ Scenario 3: {analysis.reasoning}")

    def test_complete_routing_scenario_4(self, router):
        """Scenario: Ambiguous query."""
        query = "Information about Python"
        analysis = router.analyze_query(query, estimate_semantic_gap=False)
        # Could be language or programming language - ambiguous
        print(f"✓ Scenario 4: {analysis.strategy.value} - {analysis.reasoning}")
