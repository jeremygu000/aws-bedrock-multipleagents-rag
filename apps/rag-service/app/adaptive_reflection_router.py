"""Adaptive reflection router: Decide when to reflect vs skip for cost optimization."""

from __future__ import annotations

import logging
import random

from .models import RetrieveRequest
from .self_reflection_models import AdaptiveReflectionDecision, QueryComplexityScore

logger = logging.getLogger(__name__)


class AdaptiveReflectionRouter:
    """Routes queries to reflection or skips based on complexity heuristics.
    
    Goal: Skip ~65% of simple queries to reduce cost while reflecting on complex ones.
    """

    # Keywords indicating reasoning/analytical queries
    REASONING_KEYWORDS = {
        "explain", "why", "how", "discuss", "analyze", "compare", "contrast",
        "elaborate", "describe", "interpret", "evaluate", "assess", "justify",
    }

    # Keywords indicating simple entity/factual queries
    ENTITY_KEYWORDS = {
        "who", "what is", "where", "when", "find", "list", "which",
        "tell me about", "define", "identify",
    }

    def __init__(self, enable_reflection: bool = True):
        """Initialize router."""
        self._enable_reflection = enable_reflection

    def estimate_query_complexity(
        self,
        question: str,
        request: RetrieveRequest | None = None,  # noqa: ARG001
    ) -> QueryComplexityScore:
        """Estimate query complexity for adaptive reflection routing.

        Args:
            question: User question
            request: Retrieve request with metadata (optional)

        Returns:
            QueryComplexityScore with components and final complexity level
        """
        tokens = question.split()
        token_count = len(tokens)
        question_lower = question.lower()

        # Rule 1: Check for reasoning keywords
        has_reasoning = any(kw in question_lower for kw in self.REASONING_KEYWORDS)

        # Rule 2: Check for entity query pattern
        is_entity_query = any(kw in question_lower for kw in self.ENTITY_KEYWORDS)

        # Rule 3: Check for multi-entity (compare/contrast patterns)
        is_multi_entity = any(
            p in question_lower for p in ["vs", "versus", "vs.", "compared to"]
        ) or question_lower.count(" and ") > 0

        # Rule 4: Estimate semantic gap (query vs docs)
        # High semantic gap = less obvious retrievals = needs reflection
        semantic_gap = 0.0
        if is_entity_query:
            semantic_gap = 0.1  # Low gap for direct entity queries
        elif has_reasoning:
            semantic_gap = 0.5  # Higher gap for reasoning queries
        elif token_count > 20:
            semantic_gap = 0.4  # Longer queries have larger semantic gap
        else:
            semantic_gap = 0.2  # Default moderate gap

        # Rule 5: Map to complexity level
        if token_count < 5 or (is_entity_query and token_count < 15):
            complexity = "simple"
        elif token_count > 30 or is_multi_entity or has_reasoning:
            complexity = "complex"
        else:
            complexity = "moderate"

        return QueryComplexityScore(
            token_count=token_count,
            has_reasoning_keywords=has_reasoning,
            is_entity_query=is_entity_query,
            is_multi_entity=is_multi_entity,
            semantic_gap_score=min(semantic_gap, 1.0),
            estimated_complexity=complexity,  # type: ignore
        )

    def should_reflect(
        self,
        question: str,
        answer: str,  # noqa: ARG002
        hits_count: int,
        model_confidence: float = 0.0,
        request: RetrieveRequest | None = None,
        force_reflect: bool = False,
    ) -> AdaptiveReflectionDecision:
        """Decide whether to reflect on answer based on query/answer characteristics.

        Strategy:
        - Always skip: Simple entity queries, very high model confidence
        - Always reflect: Complex reasoning, low hits, low confidence
        - Sample: Moderate complexity (30% baseline)

        Args:
            question: Original question
            answer: Generated answer to validate
            hits_count: Number of retrieved documents
            model_confidence: Model's confidence in answer (0-1)
            request: Retrieve request (for filters/context)
            force_reflect: Force reflection regardless of heuristics

        Returns:
            Decision with reasoning
        """
        decision = self._decide_reflect(
            question, hits_count, model_confidence, request, force_reflect,
        )
        logger.info(
            "ADAPTIVE_ROUTER_DECISION reflect=%s hits=%d confidence=%.2f reason=%s query=%r",
            decision.should_reflect,
            hits_count,
            model_confidence,
            decision.reason,
            question[:80],
        )
        return decision

    def _decide_reflect(
        self,
        question: str,
        hits_count: int,
        model_confidence: float,
        request: RetrieveRequest | None,
        force_reflect: bool,
    ) -> AdaptiveReflectionDecision:
        if force_reflect:
            return AdaptiveReflectionDecision(
                should_reflect=True,
                reason="Force reflection enabled",
            )

        if not self._enable_reflection:
            return AdaptiveReflectionDecision(
                should_reflect=False,
                reason="Reflection disabled globally",
            )

        complexity = self.estimate_query_complexity(question, request)

        # Decision tree
        reasons = []

        # Rule 1: Skip simple entity queries
        if complexity.is_entity_query and complexity.token_count < 15:
            reasons.append("Skip: Simple entity query")
            return AdaptiveReflectionDecision(
                should_reflect=False,
                reason=" | ".join(reasons),
            )

        # Rule 2: Skip very high confidence answers
        if model_confidence > 0.95:
            reasons.append(f"Skip: High model confidence ({model_confidence:.2f})")
            return AdaptiveReflectionDecision(
                should_reflect=False,
                reason=" | ".join(reasons),
            )

        # Rule 3: Always reflect if few hits (low retrieval quality)
        if hits_count < 3:
            reasons.append(f"Reflect: Low hit count ({hits_count})")
            return AdaptiveReflectionDecision(
                should_reflect=True,
                reason=" | ".join(reasons),
            )

        # Rule 4: Always reflect if low model confidence
        if model_confidence < 0.5:
            reasons.append(f"Reflect: Low model confidence ({model_confidence:.2f})")
            return AdaptiveReflectionDecision(
                should_reflect=True,
                reason=" | ".join(reasons),
            )

        # Rule 5: Always reflect for complex queries with high semantic gap
        if (
            complexity.estimated_complexity == "complex"
            and complexity.semantic_gap_score > 0.3
        ):
            reasons.append(
                f"Reflect: Complex query + high semantic gap ({complexity.semantic_gap_score:.2f})"
            )
            return AdaptiveReflectionDecision(
                should_reflect=True,
                reason=" | ".join(reasons),
            )

        if complexity.estimated_complexity == "moderate":
            if random.random() < 0.30:  # noqa: S311
                reasons.append("Sample: Moderate complexity query (30% baseline)")
                return AdaptiveReflectionDecision(
                    should_reflect=True,
                    reason=" | ".join(reasons),
                )
            else:
                reasons.append("Skip: Moderate complexity query (sampling missed)")
                return AdaptiveReflectionDecision(
                    should_reflect=False,
                    reason=" | ".join(reasons),
                )

        # Default: Skip (optimistic)
        reasons.append("Default: Skip (optimistic routing)")
        return AdaptiveReflectionDecision(
            should_reflect=False,
            reason=" | ".join(reasons),
        )
