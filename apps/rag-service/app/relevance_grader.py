"""Relevance grader: LLM-as-judge for answer-question alignment."""

from __future__ import annotations

import asyncio
import json
import logging

from .bedrock_embedding_client import BedrockEmbeddingClient
from .config import Settings
from .qwen_client import QwenClient
from .self_reflection_models import RelevanceResult, RelevanceVerdict

logger = logging.getLogger(__name__)


class RelevanceGrader:
    """LLM-based relevance evaluator checking answer-question alignment."""

    RELEVANCE_CHECK_PROMPT = """\
You are an expert evaluator. Your job is to assess how well an answer addresses a question.

ORIGINAL QUESTION:
{question}

GENERATED ANSWER:
{answer}

Evaluate the following dimensions (0.0-1.0):
1. **Alignment**: Does the answer address the question directly?
2. **Completeness**: Does it cover all aspects of the question?
3. **Relevance**: Are the provided details relevant and useful?

Respond in JSON format only:
{
  "alignment_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "explanation": "brief explanation"
}"""

    def __init__(
        self,
        settings: Settings,
        bedrock_client: BedrockEmbeddingClient,
        qwen_client: QwenClient | None = None,
    ):
        """Initialize grader."""
        self._settings = settings
        self._bedrock_client = bedrock_client
        self._qwen_client = qwen_client

    async def grade_relevance(
        self,
        question: str,
        answer: str,
    ) -> RelevanceResult:
        """Grade relevance of answer to question.

        Args:
            question: Original user question
            answer: Generated answer

        Returns:
            RelevanceResult with alignment, completeness, and overall scores
        """
        try:
            prompt = self.RELEVANCE_CHECK_PROMPT.format(
                question=question[:500],
                answer=answer[:1500],
            )

            if self._qwen_client and self._qwen_client.is_configured():
                response = await asyncio.to_thread(
                    self._qwen_client.chat,
                    "You are an expert evaluator. Output JSON only.",
                    prompt,
                    max_tokens=300,
                )
            else:
                response = (
                    '{"alignment_score": 0.7, "completeness_score": 0.7, '
                    '"relevance_score": 0.7, "explanation": "Default relevance"}'
                )

            # Parse JSON response
            result = json.loads(response)

            alignment = float(result.get("alignment_score", 0.5))
            completeness = float(result.get("completeness_score", 0.5))
            relevance = float(result.get("relevance_score", 0.5))

            # Overall score is average
            overall_score = (alignment + completeness + relevance) / 3.0

            # Map to verdict
            if overall_score >= 0.75:
                verdict = RelevanceVerdict.RELEVANT
            elif overall_score >= 0.5:
                verdict = RelevanceVerdict.PARTIALLY_RELEVANT
            else:
                verdict = RelevanceVerdict.IRRELEVANT

            return RelevanceResult(
                verdict=verdict,
                score=overall_score,
                alignment_score=alignment,
                completeness_score=completeness,
                explanation=result.get("explanation", ""),
            )

        except json.JSONDecodeError:
            logger.warning("Failed to parse relevance response, using default")
            return RelevanceResult(
                verdict=RelevanceVerdict.PARTIALLY_RELEVANT,
                score=0.5,
                alignment_score=0.5,
                completeness_score=0.5,
                explanation="Unable to evaluate",
            )
        except Exception as e:
            logger.error(f"Relevance grading failed: {e}")
            return RelevanceResult(
                verdict=RelevanceVerdict.PARTIALLY_RELEVANT,
                score=0.5,
                alignment_score=0.5,
                completeness_score=0.5,
                explanation=f"Grading error: {str(e)[:100]}",
            )
