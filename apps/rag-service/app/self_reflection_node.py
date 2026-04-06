"""Self-Reflection node for LangGraph: Post-generation answer validation.

Runs parallel faithfulness + relevance grading, then decides retry action.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .adaptive_reflection_router import AdaptiveReflectionRouter
from .config import Settings
from .faithfulness_grader import FaithfulnessGrader
from .relevance_grader import RelevanceGrader
from .self_reflection_models import (
    FaithfulnessResult,
    FaithfulnessVerdict,
    ReflectionResult,
    RelevanceResult,
    RelevanceVerdict,
    RetryAction,
)

logger = logging.getLogger(__name__)


class SelfReflectionNode:
    """Post-generation reflection node for hallucination detection and validation."""

    def __init__(
        self,
        settings: Settings,
        faithfulness_grader: FaithfulnessGrader,
        relevance_grader: RelevanceGrader,
        adaptive_router: AdaptiveReflectionRouter | None = None,
    ):
        """Initialize reflection node."""
        self._settings = settings
        self._faithfulness_grader = faithfulness_grader
        self._relevance_grader = relevance_grader
        self._adaptive_router = (
            adaptive_router or AdaptiveReflectionRouter(enable_reflection=True)
        )

    async def reflect_on_answer(
        self,
        question: str,
        answer: str,
        hits: list[dict[str, Any]],
        model_confidence: float = 0.0,
        force_reflect: bool = False,
    ) -> ReflectionResult:
        """Run post-generation reflection on answer.

        Process:
        1. Decide whether to reflect (adaptive routing)
        2. If yes: Run parallel faithfulness + relevance grading
        3. Aggregate results and decide retry action

        Args:
            question: Original user question
            answer: Generated answer to validate
            hits: Retrieved documents (for faithfulness grounding)
            model_confidence: Model's confidence in answer (0-1)
            force_reflect: Override adaptive routing and always reflect

        Returns:
            ReflectionResult with verdicts, scores, and retry action
        """
        start_time = time.time()

        try:
            # Step 1: Adaptive routing decision
            evidence_texts = [h.get("snippet", h.get("chunk_text", "")) for h in hits]

            router_decision = self._adaptive_router.should_reflect(
                question=question,
                answer=answer,
                hits_count=len(hits),
                model_confidence=model_confidence,
                force_reflect=force_reflect,
            )

            if not router_decision.should_reflect:
                logger.info(
                    "Reflection skipped: %s", router_decision.reason,
                )
                return self._neutral_result(router_decision.reason, latency_ms=0.0)

            # Step 2: Parallel grading (faithfulness + relevance)
            logger.debug(f"Running reflection on answer: {answer[:100]}...")

            f_task = asyncio.create_task(
                self._faithfulness_grader.grade_faithfulness(
                    answer=answer,
                    evidence_texts=evidence_texts,
                    question=question,
                )
            )
            r_task = asyncio.create_task(
                self._relevance_grader.grade_relevance(
                    question=question,
                    answer=answer,
                )
            )

            f_result, r_result = await asyncio.gather(f_task, r_task)

            # Step 3: Aggregate and decide retry action
            retry_action = self._decide_retry_action(
                f_result=f_result,
                r_result=r_result,
                hits_count=len(hits),
            )

            # Overall confidence: average of both scores
            overall_confidence = (f_result.score + r_result.score) / 2.0

            latency_ms = (time.time() - start_time) * 1000

            return ReflectionResult(
                faithfulness=f_result,
                relevance=r_result,
                should_retry=(retry_action != RetryAction.ACCEPT),
                retry_action=retry_action,
                overall_confidence=overall_confidence,
                explanation=(
                    f"Faithfulness: {f_result.verdict.value} ({f_result.score:.2f}) | "
                    f"Relevance: {r_result.verdict.value} ({r_result.score:.2f}) | "
                    f"Action: {retry_action.value}"
                ),
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error("Reflection failed: %s", e, exc_info=True)
            return self._neutral_result(
                f"Reflection error: {str(e)[:100]}",
                latency_ms=(time.time() - start_time) * 1000,
                confidence=0.5,
            )

    def _decide_retry_action(
        self,
        f_result,
        r_result,
        hits_count: int,
    ) -> RetryAction:
        """Decide what action to take based on grading results.

        Rules:
        - Both scores >= 0.7: ACCEPT (answer is good)
        - Faithfulness < 0.5: RETRY_RETRIEVAL (hallucinating)
        - Relevance < 0.5 & hits >= 3: REFINE_ANSWER (need better generation)
        - Either score < 0.4 & few hits: RETRY_RETRIEVAL (low quality evidence)
        - Else: FALLBACK_MODEL (need stronger model)
        """
        f_score = f_result.score
        r_score = r_result.score

        # Accept good answers
        if f_score >= 0.7 and r_score >= 0.7:
            return RetryAction.ACCEPT

        # Strong hallucination indicators
        if f_score < 0.5:
            logger.warning(
                f"Low faithfulness ({f_score:.2f}), retrying retrieval"
            )
            return RetryAction.RETRY_RETRIEVAL

        # Irrelevant answers with good evidence
        if r_score < 0.5 and hits_count >= 3:
            logger.warning(
                f"Low relevance ({r_score:.2f}) despite good evidence, refining answer"
            )
            return RetryAction.REFINE_ANSWER

        # Low quality overall + few hits
        if (f_score < 0.6 or r_score < 0.6) and hits_count < 3:
            logger.warning(
                f"Low scores (f={f_score:.2f}, r={r_score:.2f}) with few hits, retrying retrieval"
            )
            return RetryAction.RETRY_RETRIEVAL

        # Default: use stronger model
        logger.warning(
            f"Intermediate scores (f={f_score:.2f}, r={r_score:.2f}), using stronger model"
        )
        return RetryAction.FALLBACK_MODEL

    @staticmethod
    def _neutral_result(
        explanation: str,
        latency_ms: float = 0.0,
        confidence: float = 0.7,
    ) -> ReflectionResult:
        return ReflectionResult(
            faithfulness=FaithfulnessResult(
                verdict=FaithfulnessVerdict.FAITHFUL,
                score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                explanation="Skipped",
            ),
            relevance=RelevanceResult(
                verdict=RelevanceVerdict.RELEVANT,
                score=1.0,
                alignment_score=1.0,
                completeness_score=1.0,
                explanation="Skipped",
            ),
            should_retry=False,
            retry_action=RetryAction.ACCEPT,
            overall_confidence=confidence,
            explanation=explanation,
            latency_ms=latency_ms,
        )
