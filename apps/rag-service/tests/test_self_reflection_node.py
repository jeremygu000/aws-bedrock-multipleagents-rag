"""Tests for Self-Reflection nodes."""

from __future__ import annotations

import pytest

from app.adaptive_reflection_router import AdaptiveReflectionRouter
from app.self_reflection_models import (
    QueryComplexityScore,
)


class TestAdaptiveReflectionRouter:
    """Test adaptive reflection routing logic."""

    def test_simple_entity_query_skips_reflection(self):
        """Simple entity queries should skip reflection."""
        router = AdaptiveReflectionRouter(enable_reflection=True)

        decision = router.should_reflect(
            question="who is Adele",
            answer="Adele is a singer",
            hits_count=5,
            model_confidence=0.8,
        )

        assert decision.should_reflect is False
        assert "Simple entity query" in decision.reason

    def test_complex_reasoning_query_reflects(self):
        """Complex reasoning queries should trigger reflection."""
        router = AdaptiveReflectionRouter(enable_reflection=True)

        decision = router.should_reflect(
            question="Explain how copyright law differs between Australia and the US",
            answer="Australian copyright law... US copyright law...",
            hits_count=5,
            model_confidence=0.6,
        )

        # Complex + semantic gap > 0.3 should trigger reflection
        assert decision.should_reflect is True

    def test_low_hits_always_reflect(self):
        """Reflect when retrieval yields too few hits (non-entity query)."""
        router = AdaptiveReflectionRouter(enable_reflection=True)

        decision = router.should_reflect(
            question="Explain the difference between copyright law in Australia and the US",
            answer="I found limited information",
            hits_count=1,
            model_confidence=0.5,
        )

        assert decision.should_reflect is True
        assert "Low hit count" in decision.reason

    def test_high_confidence_skips(self):
        """Skip reflection for very high confidence answers (non-entity query)."""
        router = AdaptiveReflectionRouter(enable_reflection=True)

        decision = router.should_reflect(
            question="Summarise the latest regulatory changes in streaming royalties",
            answer="The latest regulatory changes include...",
            hits_count=10,
            model_confidence=0.99,
        )

        assert decision.should_reflect is False
        assert "High model confidence" in decision.reason

    def test_low_confidence_reflects(self):
        """Reflect for low-confidence answers."""
        router = AdaptiveReflectionRouter(enable_reflection=True)

        decision = router.should_reflect(
            question="Estimate future music streaming revenue",
            answer="Approximately 50-60 billion",
            hits_count=5,
            model_confidence=0.3,
        )

        assert decision.should_reflect is True
        assert "Low model confidence" in decision.reason

    def test_force_reflect_overrides(self):
        """Force reflection flag should override all routing logic."""
        router = AdaptiveReflectionRouter(enable_reflection=True)

        decision = router.should_reflect(
            question="who is Adele",
            answer="Adele is a singer",
            hits_count=5,
            model_confidence=0.99,
            force_reflect=True,
        )

        assert decision.should_reflect is True
        assert "Force reflection" in decision.reason

    def test_global_disable_reflection(self):
        """Global disable flag should skip all reflection."""
        router = AdaptiveReflectionRouter(enable_reflection=False)

        decision = router.should_reflect(
            question="Explain complex music licensing",
            answer="Long complex answer",
            hits_count=5,
            model_confidence=0.2,
        )

        assert decision.should_reflect is False
        assert "disabled globally" in decision.reason

    def test_query_complexity_estimation_simple(self):
        """Estimate simple query complexity correctly."""
        router = AdaptiveReflectionRouter()

        score = router.estimate_query_complexity("Who is Alice?")

        assert score.token_count == 3
        assert score.is_entity_query is True
        assert score.estimated_complexity == "simple"

    def test_query_complexity_estimation_complex(self):
        """Estimate complex query complexity correctly."""
        router = AdaptiveReflectionRouter()

        score = router.estimate_query_complexity(
            "How does copyright law differ between Australia and the US in terms of fair use provisions?"
        )

        assert score.has_reasoning_keywords is True
        assert score.semantic_gap_score > 0.3
        assert score.estimated_complexity == "complex"

    def test_query_complexity_estimation_moderate(self):
        """Estimate moderate query complexity correctly."""
        router = AdaptiveReflectionRouter()

        score = router.estimate_query_complexity(
            "What are the main revenue streams for APRA"
        )

        assert score.estimated_complexity == "moderate"


class TestReflectionModels:
    """Test reflection data models."""

    def test_query_complexity_score_should_reflect_default(self):
        """Test default reflection decision logic."""
        # Simple entity query: should not reflect
        score = QueryComplexityScore(
            token_count=5,
            has_reasoning_keywords=False,
            is_entity_query=True,
            is_multi_entity=False,
            semantic_gap_score=0.1,
            estimated_complexity="simple",  # type: ignore
        )
        assert score.should_reflect_default is False

        # Complex reasoning: should reflect
        score = QueryComplexityScore(
            token_count=30,
            has_reasoning_keywords=True,
            is_entity_query=False,
            is_multi_entity=False,
            semantic_gap_score=0.5,
            estimated_complexity="complex",  # type: ignore
        )
        assert score.should_reflect_default is True


if __name__ == "__main__":
    pytest.main([__file__])
