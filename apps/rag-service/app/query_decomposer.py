"""Query Decomposition for Complex Multi-Hop Reasoning.

This module implements query decomposition patterns from CRAG and Self-RAG research:
- Decision logic for when to decompose (semantic gap, query length, reasoning keywords)
- LLM-based sub-question generation (Nova Pro via Bedrock)
- Synchronous API matching other workflow nodes (crag.py, hyde_retriever.py)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import boto3
from pydantic import BaseModel, Field

from .config import Settings

logger = logging.getLogger(__name__)


class DecompositionDecision(BaseModel):
    """Decision result for query decomposition."""

    should_decompose: bool = Field(description="Whether to decompose this query")
    confidence: float = Field(gt=0, le=1.0, description="Confidence in decision (0-1)")
    reasoning: str = Field(description="Explanation for the decision")
    estimated_sub_questions: int = Field(ge=0, le=5, description="Estimated number of sub-questions (if decomposing)")
    decomposition_cost_cents: float = Field(
        ge=0, description="Estimated cost in cents for decomposition (if applicable)"
    )


class SubQuestion(BaseModel):
    """Generated sub-question from decomposition."""

    id: int = Field(ge=1, le=5, description="Sub-question index (1-5)")
    question: str = Field(min_length=10, max_length=500)
    focus: str = Field(description="What this sub-question focuses on")
    retrieve_strategy: str = Field(description="Suggested retrieval strategy: dense, sparse, or hybrid")


class DecompositionResult(BaseModel):
    """Complete decomposition result with sub-questions."""

    should_decompose: bool
    decision_reasoning: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    original_query: str


class QueryDecomposer:
    """Intelligently decide when to decompose and generate sub-questions."""

    # Thresholds for decomposition routing (tuned from research)
    DECOMPOSITION_MIN_TOKENS = 15
    DECOMPOSITION_SEMANTIC_GAP_THRESHOLD = 0.3
    DECOMPOSITION_CONFIDENCE_THRESHOLD = 0.6

    # Reasoning keywords that suggest decomposition
    REASONING_KEYWORDS = {
        "explain",
        "why",
        "how",
        "compare",
        "contrast",
        "analyze",
        "discuss",
        "evaluate",
        "summarize",
        "relationship",
        "connection",
        "impact",
        "implication",
        "cause",
        "effect",
        "difference",
        "similarity",
    }

    # Entity keywords that suggest NO decomposition
    ENTITY_KEYWORDS = {"who", "what is", "where is", "when was", "find", "list", "search", "lookup"}

    def __init__(self, settings: Settings, bedrock_client: Any = None):
        self._settings = settings
        self._bedrock_client = bedrock_client or boto3.client("bedrock-runtime", region_name=settings.aws_region)

    def should_decompose(
        self, query: str, complexity: str = "medium", semantic_gap: Optional[float] = None
    ) -> DecompositionDecision:
        tokens = query.split()
        token_count = len(tokens)
        query_lower = query.lower()

        logger.info(
            "Decomposition decision: query=%r (tokens=%d, complexity=%s, semantic_gap=%s)",
            query[:100], token_count, complexity, semantic_gap,
        )

        # Rule 1: Too short → skip
        if token_count < self.DECOMPOSITION_MIN_TOKENS:
            logger.info(
                "Decomposition decision: SKIP — query too short (%d tokens < %d threshold)",
                token_count, self.DECOMPOSITION_MIN_TOKENS,
            )
            return DecompositionDecision(
                should_decompose=False,
                confidence=0.85,
                reasoning=f"Query too short ({token_count} tokens < {self.DECOMPOSITION_MIN_TOKENS} threshold)",
                estimated_sub_questions=0,
                decomposition_cost_cents=0.0,
            )

        # Rule 2: Explicit entity queries → skip decomposition, use BM25
        matched_entity_kw = [kw for kw in self.ENTITY_KEYWORDS if kw in query_lower]
        if matched_entity_kw:
            logger.info(
                "Decomposition decision: SKIP — entity query detected (matched=%s)",
                matched_entity_kw,
            )
            return DecompositionDecision(
                should_decompose=False,
                confidence=0.80,
                reasoning=f"Query contains entity lookup keywords: {matched_entity_kw}",
                estimated_sub_questions=0,
                decomposition_cost_cents=0.0,
            )

        # Rule 3: Reasoning-heavy + high semantic gap → decompose
        matched_reasoning_kw = [kw for kw in self.REASONING_KEYWORDS if kw in query_lower]
        is_reasoning = bool(matched_reasoning_kw)
        gap_above_threshold = semantic_gap is None or semantic_gap > self.DECOMPOSITION_SEMANTIC_GAP_THRESHOLD

        if is_reasoning and gap_above_threshold:
            estimated_subq = 2 + (1 if token_count > 25 else 0)
            cost_cents = 0.05 + (estimated_subq * 0.02)
            logger.info(
                "Decomposition decision: DECOMPOSE — reasoning keywords=%s, semantic_gap=%s > %.2f, est_subq=%d, cost=%.4f¢",
                matched_reasoning_kw, semantic_gap, self.DECOMPOSITION_SEMANTIC_GAP_THRESHOLD,
                estimated_subq, cost_cents,
            )
            return DecompositionDecision(
                should_decompose=True,
                confidence=0.85,
                reasoning=f"Reasoning-heavy query (keywords={matched_reasoning_kw}) + semantic gap {semantic_gap or 'unknown'} > threshold",
                estimated_sub_questions=estimated_subq,
                decomposition_cost_cents=cost_cents,
            )

        # Rule 4: High complexity + long query → decompose
        if complexity == "high" and token_count > 20:
            estimated_subq = 3
            cost_cents = 0.05 + (estimated_subq * 0.02)
            logger.info(
                "Decomposition decision: DECOMPOSE — high complexity + long query (tokens=%d > 20), est_subq=%d",
                token_count, estimated_subq,
            )
            return DecompositionDecision(
                should_decompose=True,
                confidence=0.75,
                reasoning=f"High complexity + long query (token_count={token_count} > 20)",
                estimated_sub_questions=estimated_subq,
                decomposition_cost_cents=cost_cents,
            )

        # Rule 5: Default NO decomposition
        logger.info(
            "Decomposition decision: SKIP — default (reasoning_kw=%s, complexity=%s, tokens=%d)",
            matched_reasoning_kw, complexity, token_count,
        )
        return DecompositionDecision(
            should_decompose=False,
            confidence=0.70,
            reasoning=f"Default skip (reasoning_kw={matched_reasoning_kw}, complexity={complexity}, tokens={token_count})",
            estimated_sub_questions=0,
            decomposition_cost_cents=0.0,
        )

    def decompose_query(self, query: str, decision: Optional[DecompositionDecision] = None) -> DecompositionResult:
        if decision is None:
            decision = self.should_decompose(query)

        if not decision.should_decompose:
            logger.info(
                "Decomposition: not recommended (confidence=%.2f, reasoning=%s)",
                decision.confidence, decision.reasoning,
            )
            return DecompositionResult(
                should_decompose=False, decision_reasoning=decision.reasoning, original_query=query
            )

        logger.info(
            "Decomposition: proceeding with LLM generation (model=%s, est_subq=%d, confidence=%.2f, cost=%.4f¢)",
            self._settings.decomposition_model_id,
            decision.estimated_sub_questions,
            decision.confidence,
            decision.decomposition_cost_cents,
        )

        start = time.perf_counter()
        try:
            sub_questions = self._generate_sub_questions_bedrock(
                query, decision.estimated_sub_questions
            )
            elapsed = time.perf_counter() - start
            logger.info(
                "Decomposition: LLM generation completed in %.2fs — %d sub-questions generated",
                elapsed, len(sub_questions),
            )
            return DecompositionResult(
                should_decompose=True,
                decision_reasoning=decision.reasoning,
                sub_questions=sub_questions,
                original_query=query,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.exception("Decomposition: LLM generation FAILED after %.2fs: %s", elapsed, e)
            return DecompositionResult(
                should_decompose=False,
                decision_reasoning=f"Decomposition failed: {str(e)}",
                original_query=query,
            )

    def _generate_sub_questions_bedrock(self, query: str, num_subquestions: int) -> list[SubQuestion]:
        num_subquestions = max(2, min(num_subquestions, self._settings.decomposition_max_subquestions))
        model_id = self._settings.decomposition_model_id

        system_prompt = (
            "You are a query decomposer. Given a complex query, break it down into "
            f"{num_subquestions} focused sub-questions that collectively address the original query.\n\n"
            "Each sub-question should:\n"
            "- Focus on ONE specific aspect\n"
            "- Be answerable independently\n"
            "- Use simple, direct language\n"
            "- Suggest a retrieval strategy (dense, sparse, or hybrid)\n\n"
            'Output ONLY a JSON array with objects: '
            '{"id": 1, "question": "...", "focus": "...", "retrieve_strategy": "dense|sparse|hybrid"}'
        )

        user_prompt = f"Original query: {query}\n\nGenerate {num_subquestions} focused sub-questions."

        logger.info(
            "Decomposition Bedrock: calling model=%s, max_tokens=500, temperature=0.5, query=%r",
            model_id, query[:100],
        )

        try:
            start = time.perf_counter()
            response = self._bedrock_client.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={
                    "maxTokens": 500,
                    "temperature": 0.5,
                },
            )
            elapsed = time.perf_counter() - start

            text_content = ""
            for block in response.get("output", {}).get("message", {}).get("content", []):
                if "text" in block:
                    text_content += block["text"]

            usage = response.get("usage", {})
            logger.info(
                "Decomposition Bedrock: response received in %.2fs — raw_len=%d, input_tokens=%s, output_tokens=%s",
                elapsed, len(text_content),
                usage.get("inputTokens", "?"), usage.get("outputTokens", "?"),
            )
            logger.debug("Decomposition Bedrock: raw response text=%s", text_content[:500])

            return self._parse_sub_questions(text_content, query, num_subquestions)

        except Exception as e:
            logger.exception("Decomposition Bedrock: call to model=%s FAILED: %s", model_id, e)
            return self._create_fallback_subquestions(query, num_subquestions)

    def _parse_sub_questions(self, text: str, query: str, num_subquestions: int) -> list[SubQuestion]:
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]

        try:
            sub_questions_data = json.loads(text)
            if not isinstance(sub_questions_data, list):
                sub_questions_data = [sub_questions_data]
            parsed = [SubQuestion(**sq) for sq in sub_questions_data if isinstance(sq, dict)]
            if parsed:
                logger.info(
                    "Decomposition parse: SUCCESS — parsed %d/%d sub-questions from JSON",
                    len(parsed), num_subquestions,
                )
                for sq in parsed:
                    logger.info(
                        "  Sub-question [%d]: %r (focus=%s, strategy=%s)",
                        sq.id, sq.question[:80], sq.focus, sq.retrieve_strategy,
                    )
                return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "Decomposition parse: FAILED to parse JSON (%s): %s", type(e).__name__, text[:200],
            )

        logger.info("Decomposition parse: falling back to deterministic sub-questions")
        return self._create_fallback_subquestions(query, num_subquestions)

    def _create_fallback_subquestions(self, query: str, num_subquestions: int) -> list[SubQuestion]:
        logger.warning(
            "Decomposition fallback: generating %d deterministic sub-questions for query=%r",
            num_subquestions, query[:80],
        )
        questions = [
            SubQuestion(
                id=1,
                question=f"What is the context and background of: {query[:80]}",
                focus="Initial context and setup",
                retrieve_strategy="dense",
            )
        ]

        if num_subquestions >= 2:
            questions.append(
                SubQuestion(
                    id=2,
                    question=f"What are the implications of {query.lower()[:80]}?",
                    focus="Analysis and implications",
                    retrieve_strategy="dense",
                )
            )

        if num_subquestions >= 3:
            questions.append(
                SubQuestion(
                    id=3,
                    question=f"How does this relate to broader context of {query.lower()[:50]}?",
                    focus="Synthesis and broader context",
                    retrieve_strategy="hybrid",
                )
            )

        for sq in questions:
            logger.info("  Fallback sub-question [%d]: %r", sq.id, sq.question[:80])
        return questions
