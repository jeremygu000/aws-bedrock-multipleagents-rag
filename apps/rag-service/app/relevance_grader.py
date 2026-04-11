"""Relevance grader: LLM-as-judge for answer-question alignment.

All LLM calls use Bedrock Nova Pro via the converse API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import boto3

from .config import Settings
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
{{
  "alignment_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "explanation": "brief explanation"
}}"""

    GRADER_MODEL_ID = "amazon.nova-pro-v1:0"

    def __init__(
        self,
        settings: Settings,
    ):
        """Initialize grader."""
        self._settings = settings
        self._bedrock_client: Any = None

    def _get_bedrock_client(self) -> Any:
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self._settings.aws_region
            )
        return self._bedrock_client

    def _converse(self, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
        client = self._get_bedrock_client()
        response = client.converse(
            modelId=self._settings.grader_model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]

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

            response = await asyncio.to_thread(
                self._converse,
                "You are an expert evaluator. Output JSON only.",
                prompt,
                300,
            )

            # Strip markdown code fences if present
            cleaned = response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            result = json.loads(cleaned)

            alignment = float(result.get("alignment_score", 0.5))
            completeness = float(result.get("completeness_score", 0.5))
            relevance = float(result.get("relevance_score", 0.5))

            overall_score = (alignment + completeness + relevance) / 3.0

            if overall_score >= 0.75:
                verdict = RelevanceVerdict.RELEVANT
            elif overall_score >= 0.5:
                verdict = RelevanceVerdict.PARTIALLY_RELEVANT
            else:
                verdict = RelevanceVerdict.IRRELEVANT

            logger.info(
                "RELEVANCE_GRADE verdict=%s score=%.2f "
                "alignment=%.2f completeness=%.2f relevance=%.2f",
                verdict.value,
                overall_score,
                alignment,
                completeness,
                relevance,
            )

            return RelevanceResult(
                verdict=verdict,
                score=overall_score,
                alignment_score=alignment,
                completeness_score=completeness,
                explanation=result.get("explanation", ""),
            )

        except json.JSONDecodeError:
            # Fail OPEN: default to partially relevant on parse error
            logger.warning("Failed to parse relevance response — failing OPEN (partially_relevant)")
            return RelevanceResult(
                verdict=RelevanceVerdict.PARTIALLY_RELEVANT,
                score=0.6,
                alignment_score=0.6,
                completeness_score=0.6,
                explanation="JSON parse error, defaulting to partially relevant",
            )
        except Exception as e:
            logger.error(f"Relevance grading failed: {e}")
            return RelevanceResult(
                verdict=RelevanceVerdict.PARTIALLY_RELEVANT,
                score=0.6,
                alignment_score=0.6,
                completeness_score=0.6,
                explanation=f"Grading error: {str(e)[:100]}",
            )
