"""
Query Router for adaptive HyDE + BM25 hybrid retrieval.

Intelligently routes queries to HyDE (for semantic search) or BM25 (for exact/entity queries)
based on query characteristics, semantic gap, and explicit routing rules.
"""

import logging
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """Retrieval strategy options."""

    HYDE_PRIMARY = "hyde_primary"  # HyDE with reranking
    BM25_PRIMARY = "bm25_primary"  # BM25 for exact/entity queries
    HYBRID = "hybrid"  # Both HyDE and BM25, fuse results


class QueryAnalysis(BaseModel):
    """Query analysis result."""

    strategy: RetrievalStrategy
    confidence: float = Field(gt=0, le=1.0, description="Confidence in strategy choice (0-1)")
    reasoning: str = Field(description="Explanation for routing decision")
    has_entities: bool = Field(description="Contains named entities")
    is_factual: bool = Field(description="Factual/lookup query (not reasoning)")
    is_reasoning: bool = Field(description="Reasoning-heavy query")
    query_tokens: int = Field(description="Number of tokens in query")
    semantic_gap: Optional[float] = Field(
        default=None, description="Estimated semantic gap score (higher = more gap)"
    )


class QueryRouter:
    """Route queries to optimal retrieval strategy."""

    def __init__(self, bm25_ranker=None, embeddings=None):
        """Initialize query router.

        Args:
            bm25_ranker: Optional BM25 ranker for scoring
            embeddings: Optional embedding model for semantic gap estimation
        """
        self.bm25 = bm25_ranker
        self.embeddings = embeddings

    def analyze_query(self, query: str, estimate_semantic_gap: bool = True) -> QueryAnalysis:
        """Analyze query and recommend retrieval strategy.

        Args:
            query: Query string
            estimate_semantic_gap: Whether to estimate semantic gap via BM25

        Returns:
            QueryAnalysis with recommended strategy
        """
        tokens = query.split()
        token_count = len(tokens)

        # Feature extraction
        has_entities = self._detect_entities(query)
        is_factual = self._is_factual_query(query)
        is_reasoning = self._is_reasoning_query(query)
        semantic_gap = None

        # Semantic gap estimation (if BM25 available)
        if estimate_semantic_gap and self.bm25:
            semantic_gap = self._estimate_semantic_gap(query)

        # Routing logic
        strategy, confidence, reasoning = self._route(
            token_count=token_count,
            has_entities=has_entities,
            is_factual=is_factual,
            is_reasoning=is_reasoning,
            semantic_gap=semantic_gap,
        )

        analysis = QueryAnalysis(
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            has_entities=has_entities,
            is_factual=is_factual,
            is_reasoning=is_reasoning,
            query_tokens=token_count,
            semantic_gap=semantic_gap,
        )

        logger.info(
            f"Query routing: {strategy.value} (confidence={confidence:.2f}) | "
            f"tokens={token_count}, entities={has_entities}, "
            f"factual={is_factual}, reasoning={is_reasoning}"
        )

        return analysis

    def _detect_entities(self, query: str) -> bool:
        """Detect named entities (proper nouns, dates, IDs).

        Returns:
            True if query contains likely named entities
        """
        # Capitalized proper nouns (2+ words)
        caps_words = len(re.findall(r"\b[A-Z][a-z]+\b", query))
        if caps_words >= 2:
            return True

        # Dates (YYYY or MM/DD)
        if re.search(r"\b(19|20)\d{2}\b|\b(0?[1-9]|1[0-2])/\d{1,2}\b", query):
            return True

        # IDs/codes (alphanumeric + hyphens)
        if re.search(r"\b[A-Z]{2,}-\d+\b|\bID:\s*\d+\b|\bWINF\w+\b", query):
            return True

        # Numbers (standalone)
        numbers = len(re.findall(r"\b\d+\b", query))
        if numbers > 2:
            return True

        return False

    def _is_factual_query(self, query: str) -> bool:
        """Detect factual/lookup queries (vs reasoning).

        Returns:
            True if query is primarily factual lookup
        """
        factual_keywords = [
            "what",
            "when",
            "where",
            "who",
            "which",
            "how many",
            "how much",
            "list",
            "find",
            "search",
            "lookup",
            "get",
            "retrieve",
            "show me",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in factual_keywords)

    def _is_reasoning_query(self, query: str) -> bool:
        """Detect reasoning-heavy queries.

        Returns:
            True if query requires reasoning
        """
        reasoning_keywords = [
            "explain",
            "why",
            "how does",
            "compare",
            "contrast",
            "analyze",
            "discuss",
            "evaluate",
            "summarize",
            "implications",
            "relationship",
            "connection",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in reasoning_keywords)

    def _estimate_semantic_gap(self, query: str) -> float:
        """Estimate semantic gap between query and documents (0-1).

        Higher gap = more benefit from HyDE.
        Lower gap = BM25 sufficient.

        Args:
            query: Query string

        Returns:
            Semantic gap score (0-1), or 0.5 if estimation fails
        """
        if not self.bm25:
            return 0.5  # Default neutral

        try:
            # Tokenize query
            tokens = query.lower().split()
            tokens = [t for t in tokens if len(t) > 2]  # Remove stopwords

            # If too few unique tokens, gap is low (BM25 will work)
            if len(tokens) < 3:
                return 0.3

            # If query has many rare/specific terms, gap is high (HyDE helps)
            # This is a heuristic: long queries with specific terms benefit from HyDE
            avg_token_length = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
            specific_term_ratio = len([t for t in tokens if len(t) > 5]) / len(tokens) if tokens else 0

            # Gap increases with specificity and query length
            gap_score = min(0.9, 0.2 + (len(tokens) / 20) + (specific_term_ratio * 0.3))
            return gap_score

        except Exception as e:
            logger.warning(f"Failed to estimate semantic gap: {e}")
            return 0.5

    def _route(
        self, token_count: int, has_entities: bool, is_factual: bool, is_reasoning: bool, semantic_gap: Optional[float]
    ) -> tuple[RetrievalStrategy, float, str]:
        """Route query to strategy.

        Args:
            token_count: Query token count
            has_entities: Query contains named entities
            is_factual: Query is factual
            is_reasoning: Query requires reasoning
            semantic_gap: Estimated semantic gap (None if not available)

        Returns:
            (strategy, confidence, reasoning)
        """
        # Rule 1: Too short → BM25 (cost > benefit)
        if token_count < 5:
            return RetrievalStrategy.BM25_PRIMARY, 0.85, "Query too short for HyDE"

        # Rule 2: Named entities → BM25 (HyDE hallucinates numbers/names)
        if has_entities and token_count < 15:
            return RetrievalStrategy.BM25_PRIMARY, 0.80, "Query contains named entities, HyDE may hallucinate"

        # Rule 3: Reasoning query → HyDE (semantic gap likely)
        if is_reasoning and token_count > 8:
            return RetrievalStrategy.HYDE_PRIMARY, 0.85, "Reasoning-heavy query benefits from HyDE"

        # Rule 4: Semantic gap estimation (if available)
        if semantic_gap is not None:
            if semantic_gap > 0.7:
                return RetrievalStrategy.HYDE_PRIMARY, 0.75, f"High semantic gap detected ({semantic_gap:.2f})"
            elif semantic_gap < 0.3:
                return RetrievalStrategy.BM25_PRIMARY, 0.70, f"Low semantic gap detected ({semantic_gap:.2f})"
            else:  # 0.3 < semantic_gap < 0.7
                return RetrievalStrategy.HYBRID, 0.65, f"Ambiguous semantic gap ({semantic_gap:.2f}), using hybrid"

        # Rule 5: Default based on factuality
        if is_factual and not is_reasoning:
            return RetrievalStrategy.BM25_PRIMARY, 0.60, "Factual query, defaulting to BM25"

        # Rule 6: Default for long queries → HyDE
        if token_count > 15:
            return RetrievalStrategy.HYDE_PRIMARY, 0.70, "Long query, defaulting to HyDE"

        # Rule 7: Catch-all hybrid
        return RetrievalStrategy.HYBRID, 0.50, "Default hybrid routing"

    def should_use_hyde(self, query: str) -> bool:
        """Quick check: should HyDE be enabled?

        Args:
            query: Query string

        Returns:
            True if HyDE should be used
        """
        analysis = self.analyze_query(query, estimate_semantic_gap=True)
        return analysis.strategy in (RetrievalStrategy.HYDE_PRIMARY, RetrievalStrategy.HYBRID)

    def should_use_bm25(self, query: str) -> bool:
        """Quick check: should BM25 be enabled?

        Args:
            query: Query string

        Returns:
            True if BM25 should be used
        """
        analysis = self.analyze_query(query, estimate_semantic_gap=True)
        return analysis.strategy in (RetrievalStrategy.BM25_PRIMARY, RetrievalStrategy.HYBRID)
