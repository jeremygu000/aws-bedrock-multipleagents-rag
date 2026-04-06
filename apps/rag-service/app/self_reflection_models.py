"""Data models for Self-Reflection nodes (LLM-as-judge)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FaithfulnessVerdict(str, Enum):
    """Faithfulness judgment from LLM-as-judge."""
    FAITHFUL = "faithful"
    PARTIALLY_FAITHFUL = "partially_faithful"
    UNFAITHFUL = "unfaithful"
    UNCLEAR = "unclear"


class RelevanceVerdict(str, Enum):
    """Relevance judgment from LLM-as-judge."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


class RetryAction(str, Enum):
    """Action to take after reflection."""
    ACCEPT = "accept"  # Answer is good, proceed to output
    RETRY_RETRIEVAL = "retry_retrieval"  # Retrieve again with rewritten query
    REFINE_ANSWER = "refine_answer"  # Re-generate with better prompt
    FALLBACK_MODEL = "fallback_model"  # Use stronger model (Claude)


class Claim(BaseModel):
    """A single claim extracted from an LLM answer."""
    text: str
    supported: bool = False
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FaithfulnessResult(BaseModel):
    """Result of faithfulness evaluation."""
    verdict: FaithfulnessVerdict
    score: float = Field(ge=0.0, le=1.0)  # (supported_claims / total_claims)
    claims: list[Claim] = Field(default_factory=list)
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    explanation: str = ""


class RelevanceResult(BaseModel):
    """Result of relevance evaluation."""
    verdict: RelevanceVerdict
    score: float = Field(ge=0.0, le=1.0)  # 0-1 relevance rating
    alignment_score: float = Field(ge=0.0, le=1.0)  # How well answer addresses question
    completeness_score: float = Field(ge=0.0, le=1.0)  # Coverage of question aspects
    explanation: str = ""


class ReflectionResult(BaseModel):
    """Combined reflection judgment from parallel faithfulness + relevance graders."""
    faithfulness: FaithfulnessResult
    relevance: RelevanceResult
    should_retry: bool
    retry_action: RetryAction
    overall_confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""
    latency_ms: float = 0.0  # End-to-end reflection time


class AdaptiveReflectionDecision(BaseModel):
    """Decision to reflect or skip based on query/answer characteristics."""
    should_reflect: bool
    reason: str
    confidence_threshold: float = Field(default=0.6)  # Minimum confidence to skip
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryComplexityScore(BaseModel):
    """Estimated query complexity for adaptive reflection routing."""
    token_count: int
    has_reasoning_keywords: bool
    is_entity_query: bool  # Names, IDs, specific facts
    is_multi_entity: bool  # Compare, contrast, multiple entities
    semantic_gap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_complexity: Literal["simple", "moderate", "complex"] = "moderate"

    @property
    def should_reflect_default(self) -> bool:
        """Default reflection decision based on query characteristics."""
        # Skip reflection for simple entity queries or very simple questions
        if self.is_entity_query and self.token_count < 15:
            return False
        # Reflect for complex reasoning or multi-entity questions
        if (self.estimated_complexity == "complex" and self.semantic_gap_score > 0.3):
            return True
        # Default: reflect for moderate/complex, skip for simple
        return self.estimated_complexity in ("moderate", "complex")
