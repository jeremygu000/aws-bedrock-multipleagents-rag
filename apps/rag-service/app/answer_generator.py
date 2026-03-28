"""Answer generation strategies and model routing for RAG responses."""

from __future__ import annotations

from typing import Any, Literal

import boto3

from .config import Settings
from .qwen_client import QwenClient

ModelRoute = Literal["nova-lite", "qwen-plus"]

_GENERIC_SYSTEM_PROMPT = (
    "You are a grounded enterprise RAG assistant. "
    "Answer ONLY using the provided evidence. "
    "Do NOT fabricate facts or add information beyond what is given. "
    "Use citation markers [1], [2], ... that map to evidence entries. "
    "Structure your response: first summarize the key finding, then provide supporting details with citations."
)

_INTENT_SYSTEM_PROMPTS: dict[str, str] = {
    "factual": (
        "You are a grounded enterprise RAG assistant. "
        "Answer ONLY using the provided evidence. "
        "Be precise and concise. "
        "Cite specific facts with [N] markers. "
        "If the evidence doesn't contain the answer, say so."
    ),
    "analytical": (
        "You are a grounded enterprise RAG assistant. "
        "Analyze the evidence to answer this question. "
        "Structure your response: "
        "1) Key finding, "
        "2) Supporting analysis with citations [N], "
        "3) Caveats or limitations based on available evidence."
    ),
    "procedural": (
        "You are a grounded enterprise RAG assistant. "
        "Provide step-by-step instructions based on the evidence. "
        "Number each step clearly. "
        "Cite the source for each step with [N] markers."
    ),
    "comparison": (
        "You are a grounded enterprise RAG assistant. "
        "Compare the items using evidence provided. "
        "Structure as: "
        "1) Summary of key differences, "
        "2) Detailed comparison with citations [N], "
        "3) Recommendation if evidence supports one."
    ),
}


def _get_intent_system_prompt(intent: str, settings: Settings) -> str:
    """Return system prompt based on intent and feature flag.

    Args:
        intent: Detected query intent (factual, analytical, procedural, comparison, other).
        settings: Runtime settings controlling feature flags.

    Returns:
        System prompt string appropriate for the intent.
    """
    if not settings.enable_intent_aware_prompts:
        return _GENERIC_SYSTEM_PROMPT
    return _INTENT_SYSTEM_PROMPTS.get(intent, _GENERIC_SYSTEM_PROMPT)


class BedrockConverseAnswerGenerator:
    """Generate grounded answers with Amazon Bedrock Runtime `converse`."""

    def __init__(self, settings: Settings) -> None:
        """Initialize with lazy Bedrock Runtime client creation."""

        self._settings = settings
        self._client: Any | None = None

    def generate(
        self,
        query: str,
        hits: list[dict[str, Any]],
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
    ) -> str:
        """Generate citation-aware answer text using Bedrock model."""

        if not hits:
            return (
                "I could not find grounded passages for this query in the current knowledge base."
            )

        context_block = _build_context_block(
            hits,
            max_chars=self._settings.answer_evidence_max_chars,
            include_scores=self._settings.enable_relevance_scores_in_evidence,
        )
        system_prompt = _get_intent_system_prompt(intent, self._settings)
        keyword_line = ""
        if keywords:
            keyword_line = f"Key topics: {', '.join(keywords)}\n"
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"{keyword_line}"
            f"Evidence:\n{context_block}\n\n"
            "Write a concise answer with clear citation markers."
        )

        client = self._get_client()
        response = client.converse(
            modelId=self._settings.answer_model_id,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": self._settings.answer_max_tokens,
                "temperature": self._settings.answer_temperature,
            },
        )
        answer = _extract_bedrock_text(response)
        if not answer:
            return "I could not produce a grounded answer from the current evidence."
        return answer

    def _get_client(self) -> Any:
        """Create Bedrock Runtime client lazily to avoid import-time failures."""

        if self._client is not None:
            return self._client
        if not self._settings.aws_region:
            raise ValueError(
                "AWS region is not configured. Set RAG_AWS_REGION, AWS_REGION, or AWS_DEFAULT_REGION."
            )
        self._client = boto3.client("bedrock-runtime", region_name=self._settings.aws_region)
        return self._client


class QwenAnswerGenerator:
    """Generate grounded answers with Qwen chat completions."""

    def __init__(self, qwen_client: QwenClient, settings: Settings) -> None:
        """Store configured Qwen client instance and settings."""

        self._qwen = qwen_client
        self._settings = settings

    def is_available(self) -> bool:
        """Return whether Qwen generator is available in this environment."""

        return self._qwen.is_configured()

    def generate(
        self,
        query: str,
        hits: list[dict[str, Any]],
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
    ) -> str:
        """Generate citation-aware answer text using Qwen."""

        if not hits:
            return (
                "I could not find grounded passages for this query in the current knowledge base."
            )

        context_block = _build_context_block(
            hits,
            max_chars=self._settings.answer_evidence_max_chars,
            include_scores=self._settings.enable_relevance_scores_in_evidence,
        )
        system_prompt = _get_intent_system_prompt(intent, self._settings)
        keyword_line = ""
        if keywords:
            keyword_line = f"Key topics: {', '.join(keywords)}\n"
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"{keyword_line}"
            f"Evidence:\n{context_block}\n\n"
            "Write a concise answer with clear citation markers."
        )
        answer = self._qwen.chat(system_prompt, user_prompt).strip()
        if not answer:
            return "I could not produce a grounded answer from the current evidence."
        return answer


class RoutedAnswerGenerator:
    """Route answer generation between Nova Lite and Qwen Plus."""

    def __init__(
        self,
        bedrock_generator: BedrockConverseAnswerGenerator,
        qwen_generator: QwenAnswerGenerator,
    ) -> None:
        """Store both generation backends for runtime routing."""

        self._bedrock = bedrock_generator
        self._qwen = qwen_generator

    def generate(
        self,
        query: str,
        hits: list[dict[str, Any]],
        preferred_model: ModelRoute,
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
    ) -> tuple[str, ModelRoute]:
        """Generate answer using preferred model with automatic fallback.

        Routing behavior:
        - If preferred model is `qwen-plus`, try Qwen first then Bedrock fallback.
        - If preferred model is `nova-lite`, try Bedrock first then Qwen fallback.
        """

        if preferred_model == "qwen-plus" and self._qwen.is_available():
            try:
                return (
                    self._qwen.generate(
                        query, hits, intent=intent, complexity=complexity, keywords=keywords
                    ),
                    "qwen-plus",
                )
            except Exception:
                # Fallback to Bedrock if Qwen call fails at runtime.
                return (
                    self._bedrock.generate(
                        query, hits, intent=intent, complexity=complexity, keywords=keywords
                    ),
                    "nova-lite",
                )

        if preferred_model == "nova-lite":
            try:
                return (
                    self._bedrock.generate(
                        query, hits, intent=intent, complexity=complexity, keywords=keywords
                    ),
                    "nova-lite",
                )
            except Exception:
                if self._qwen.is_available():
                    return (
                        self._qwen.generate(
                            query, hits, intent=intent, complexity=complexity, keywords=keywords
                        ),
                        "qwen-plus",
                    )
                raise

        # If Qwen is preferred but unavailable, default to Bedrock.
        return (
            self._bedrock.generate(
                query, hits, intent=intent, complexity=complexity, keywords=keywords
            ),
            "nova-lite",
        )


def _build_context_block(
    hits: list[dict[str, Any]],
    max_chars: int = 800,
    include_scores: bool = False,
) -> str:
    """Build formatted evidence context block for answer prompts.

    Args:
        hits: Retrieved document hits with citation and chunk_text fields.
        max_chars: Maximum characters to include per evidence chunk.
        include_scores: When True, include the relevance score in each block.

    Returns:
        Formatted multi-section string with evidence blocks.
    """
    sections: list[str] = []
    for index, hit in enumerate(hits, start=1):
        citation = hit["citation"]
        score_line = ""
        if include_scores:
            score = hit.get("score", 0.0)
            score_line = f"Relevance: {float(score):.3f}\n"
        section = (
            f"--- Evidence [{index}] ---\n"
            f"Title: {citation['title']}\n"
            f"URL: {citation['url']}\n"
            f"Date: {citation['year']}-{citation['month']:02d}\n"
            f"{score_line}"
            f"Content:\n{hit['chunk_text'][:max_chars]}"
        )
        sections.append(section)
    return "\n\n".join(sections)


def _extract_bedrock_text(response: dict[str, Any]) -> str:
    """Extract plain text from Bedrock `converse` response payload."""

    output = response.get("output", {})
    message = output.get("message", {})
    content = message.get("content", [])
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                parts.append(text_value.strip())
    return "\n".join(parts).strip()
