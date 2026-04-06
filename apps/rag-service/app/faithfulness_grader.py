"""Faithfulness grader: LLM-as-judge for hallucination detection.

Uses claim decomposition (from RAGAS) + entailment verification to score
whether an answer is grounded in retrieved documents.
"""

from __future__ import annotations

import asyncio
import json
import logging

from .bedrock_embedding_client import BedrockEmbeddingClient
from .config import Settings
from .qwen_client import QwenClient
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
{
  "supported": true|false,  // true if evidence supports the claim
  "confidence": 0.0-1.0,    // your confidence in this judgment
  "explanation": "brief explanation"
}

JSON response only (no markdown, no extra text):"""

    def __init__(
        self,
        settings: Settings,
        bedrock_client: BedrockEmbeddingClient,
        qwen_client: QwenClient | None = None,
    ):
        """Initialize grader with Bedrock + Qwen clients."""
        self._settings = settings
        self._bedrock_client = bedrock_client
        self._qwen_client = qwen_client

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
        """Extract atomic claims from answer using LLM."""
        try:
            prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer[:2000])

            if self._qwen_client and self._qwen_client.is_configured():
                # Use Qwen for fast claim extraction (sync chat wrapped for async)
                text = await asyncio.to_thread(
                    self._qwen_client.chat,
                    "You extract atomic claims from text.",
                    prompt,
                    max_tokens=500,
                )
            else:
                # Fallback: use answer itself as a single claim
                logger.debug("Qwen not configured, using answer as single claim")
                text = "CLAIM: " + answer[:200]

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
        """Verify if a claim is supported by evidence."""
        try:
            prompt = self.CLAIM_VERIFICATION_PROMPT.format(
                claim=claim[:500],
                evidence=evidence[:3000],
            )

            if self._qwen_client and self._qwen_client.is_configured():
                response = await asyncio.to_thread(
                    self._qwen_client.chat,
                    "You are a strict fact-checker. Output JSON only.",
                    prompt,
                    max_tokens=200,
                )
            else:
                response = '{"supported": false, "confidence": 0.5, "explanation": "Unable to verify"}'

            # Parse JSON response
            result = json.loads(response)
            return Claim(
                text=claim,
                supported=result.get("supported", False),
                confidence=result.get("confidence", 0.5),
            )

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse verification response for claim: {claim}")
            return Claim(text=claim, supported=False, confidence=0.0)
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return Claim(text=claim, supported=False, confidence=0.0)
