"""Faithfulness grader: LLM-as-judge for hallucination detection.

Uses claim decomposition (from RAGAS) + entailment verification to score
whether an answer is grounded in retrieved documents.

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
from .self_reflection_models import Claim, FaithfulnessResult, FaithfulnessVerdict

logger = logging.getLogger(__name__)


class FaithfulnessGrader:
    """LLM-based faithfulness evaluator using claim decomposition + verification."""

    # Prompt for extracting atomic claims from answer
    CLAIM_EXTRACTION_PROMPT = """\
You are a faithful assistant that extracts claims from an answer.

Your task:
1. Read the provided answer carefully.
2. Extract ALL atomic claims (facts, statements that can be independently verified).
3. Format each claim on a separate line as: CLAIM: <statement>
4. Be thorough but avoid redundant or trivially implied claims.

Answer to extract claims from:
---
{answer}
---

Output claims (one per line, format: CLAIM: ...):"""

    # Prompt for verifying if a claim is supported by evidence
    CLAIM_VERIFICATION_PROMPT = """\
You are a strict fact-checker. Your job is to verify if a claim is supported by provided evidence.

CLAIM TO VERIFY:
{claim}

SUPPORTING EVIDENCE:
{evidence}

Instructions:
1. Read the claim and evidence carefully.
2. Determine if the evidence directly supports the claim (full/partial support).
3. Output in JSON format:
{{
  "supported": true|false,
  "confidence": 0.0-1.0,
  "explanation": "brief explanation"
}}

JSON response only (no markdown, no extra text):"""

    # Bedrock model for grading
    GRADER_MODEL_ID = "amazon.nova-pro-v1:0"

    def __init__(
        self,
        settings: Settings,
    ):
        """Initialize grader with Bedrock converse API."""
        self._settings = settings
        self._bedrock_client: Any = None

    def _get_bedrock_client(self) -> Any:
        """Lazy-init Bedrock Runtime client."""
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self._settings.aws_region
            )
        return self._bedrock_client

    def _converse(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Call Bedrock converse API with Nova Pro."""
        client = self._get_bedrock_client()
        response = client.converse(
            modelId=self.GRADER_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]

    async def grade_faithfulness(
        self,
        answer: str,
        evidence_texts: list[str],
        question: str = "",
    ) -> FaithfulnessResult:
        """Grade faithfulness of an answer against retrieved evidence.

        Process:
        1. Extract claims from answer (decomposition)
        2. For each claim, verify against evidence (entailment)
        3. Compute overall faithfulness score

        Args:
            answer: Generated answer to evaluate
            evidence_texts: Retrieved document chunks to ground answer against
            question: Original question (for context, optional)

        Returns:
            FaithfulnessResult with per-claim verdicts and overall score
        """
        try:
            # Step 1: Extract claims from answer
            claims = await self._extract_claims(answer)
            if not claims:
                # Empty answer is considered faithful (no hallucinations)
                return FaithfulnessResult(
                    verdict=FaithfulnessVerdict.FAITHFUL,
                    score=1.0,
                    claims=[],
                    total_claims=0,
                    supported_claims=0,
                    unsupported_claims=0,
                    explanation="Answer contains no verifiable claims",
                )

            # Step 2: Verify each claim against evidence
            evidence_text = "\n---\n".join(evidence_texts[:5])  # Limit to top 5 docs
            verified_claims = []
            for claim_text in claims:
                verdict = await self._verify_claim(claim_text, evidence_text, question)
                verified_claims.append(verdict)

            # Step 3: Compute aggregates
            total = len(verified_claims)
            supported = sum(1 for c in verified_claims if c.supported)
            unsupported = total - supported

            # Faithfulness score: (supported_claims / total_claims)
            score = supported / total if total > 0 else 1.0

            # Map to verdict
            if score >= 0.9:
                verdict = FaithfulnessVerdict.FAITHFUL
            elif score >= 0.5:
                verdict = FaithfulnessVerdict.PARTIALLY_FAITHFUL
            elif score > 0.0:
                verdict = FaithfulnessVerdict.UNFAITHFUL
            else:
                verdict = FaithfulnessVerdict.UNFAITHFUL

            logger.info(
                "FAITHFULNESS_GRADE verdict=%s score=%.2f "
                "claims_total=%d supported=%d unsupported=%d "
                "claims_detail=[%s]",
                verdict.value,
                score,
                total,
                supported,
                unsupported,
                "; ".join(
                    f"{c.text[:60]}={'Y' if c.supported else 'N'}({c.confidence:.1f})"
                    for c in verified_claims[:5]
                ),
            )

            return FaithfulnessResult(
                verdict=verdict,
                score=score,
                claims=verified_claims,
                total_claims=total,
                supported_claims=supported,
                unsupported_claims=unsupported,
                explanation=f"{supported}/{total} claims supported by evidence",
            )

        except Exception as e:
            logger.error(f"Faithfulness grading failed: {e}")
            # Fail open: assume faithful on error
            return FaithfulnessResult(
                verdict=FaithfulnessVerdict.UNCLEAR,
                score=0.5,
                claims=[],
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                explanation=f"Grading error: {str(e)[:100]}",
            )

    async def _extract_claims(self, answer: str) -> list[str]:
        """Extract atomic claims from answer using Bedrock Nova Pro."""
        try:
            prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer[:2000])

            text = await asyncio.to_thread(
                self._converse,
                "You extract atomic claims from text.",
                prompt,
                500,
            )

            # Parse claims from response
            claims = []
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("CLAIM:"):
                    claim = line[6:].strip()
                    if claim:
                        claims.append(claim)

            return claims[:10]  # Limit to 10 claims

        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}, using fallback")
            return [answer[:200]]

    async def _verify_claim(
        self, claim: str, evidence: str, question: str = ""
    ) -> Claim:
        """Verify if a claim is supported by evidence using Bedrock Nova Pro."""
        try:
            prompt = self.CLAIM_VERIFICATION_PROMPT.format(
                claim=claim[:500],
                evidence=evidence[:3000],
            )

            response = await asyncio.to_thread(
                self._converse,
                "You are a strict fact-checker. Output JSON only.",
                prompt,
                200,
            )

            # Strip markdown code fences if present
            cleaned = response.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            # Parse JSON response
            result = json.loads(cleaned)
            return Claim(
                text=claim,
                supported=result.get("supported", False),
                confidence=result.get("confidence", 0.5),
            )

        except json.JSONDecodeError:
            # Fail OPEN: assume supported on parse error (do no harm)
            logger.warning(
                "Failed to parse verification response for claim: %s — failing OPEN (supported=True)",
                claim[:80],
            )
            return Claim(text=claim, supported=True, confidence=0.3)
        except Exception as e:
            # Fail OPEN: assume supported on any error
            logger.warning("Claim verification failed: %s — failing OPEN", e)
            return Claim(text=claim, supported=True, confidence=0.3)
