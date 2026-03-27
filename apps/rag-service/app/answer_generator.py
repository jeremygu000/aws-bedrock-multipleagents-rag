"""Answer generation strategies and model routing for RAG responses."""

from __future__ import annotations

from typing import Any, Literal

import boto3

from .config import Settings
from .qwen_client import QwenClient

ModelRoute = Literal["nova-lite", "qwen-plus"]


class BedrockConverseAnswerGenerator:
    """Generate grounded answers with Amazon Bedrock Runtime `converse`."""

    def __init__(self, settings: Settings) -> None:
        """Initialize with lazy Bedrock Runtime client creation."""

        self._settings = settings
        self._client: Any | None = None

    def generate(self, query: str, hits: list[dict[str, Any]]) -> str:
        """Generate citation-aware answer text using Bedrock model."""

        if not hits:
            return (
                "I could not find grounded passages for this query in the current knowledge base."
            )

        context_block = _build_context_block(hits)
        system_prompt = (
            "You are a grounded enterprise RAG assistant. "
            "Answer ONLY using the provided evidence. "
            "Do NOT fabricate facts or add information beyond what is given. "
            "Use citation markers [1], [2], ... that map to evidence entries. "
            "Structure your response: first summarize the key finding, then provide supporting details with citations."
        )
        user_prompt = (
            f"User query:\n{query}\n\n"
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

    def __init__(self, qwen_client: QwenClient) -> None:
        """Store configured Qwen client instance."""

        self._qwen = qwen_client

    def is_available(self) -> bool:
        """Return whether Qwen generator is available in this environment."""

        return self._qwen.is_configured()

    def generate(self, query: str, hits: list[dict[str, Any]]) -> str:
        """Generate citation-aware answer text using Qwen."""

        if not hits:
            return (
                "I could not find grounded passages for this query in the current knowledge base."
            )

        context_block = _build_context_block(hits)
        system_prompt = (
            "You are a grounded enterprise RAG assistant. "
            "Answer ONLY using the provided evidence. "
            "Do NOT fabricate facts or add information beyond what is given. "
            "Use citation markers [1], [2], ... that map to evidence entries. "
            "Structure your response: first summarize the key finding, then provide supporting details with citations."
        )
        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Evidence:\n{context_block}\n\n"
            "Write a concise answer with citation markers."
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
    ) -> tuple[str, ModelRoute]:
        """Generate answer using preferred model with automatic fallback.

        Routing behavior:
        - If preferred model is `qwen-plus`, try Qwen first then Bedrock fallback.
        - If preferred model is `nova-lite`, try Bedrock first then Qwen fallback.
        """

        if preferred_model == "qwen-plus" and self._qwen.is_available():
            try:
                return self._qwen.generate(query, hits), "qwen-plus"
            except Exception:
                # Fallback to Bedrock if Qwen call fails at runtime.
                return self._bedrock.generate(query, hits), "nova-lite"

        if preferred_model == "nova-lite":
            try:
                return self._bedrock.generate(query, hits), "nova-lite"
            except Exception:
                if self._qwen.is_available():
                    return self._qwen.generate(query, hits), "qwen-plus"
                raise

        # If Qwen is preferred but unavailable, default to Bedrock.
        return self._bedrock.generate(query, hits), "nova-lite"


def _build_context_block(hits: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, hit in enumerate(hits, start=1):
        citation = hit["citation"]
        section = (
            f"--- Evidence [{index}] ---\n"
            f"Title: {citation['title']}\n"
            f"URL: {citation['url']}\n"
            f"Date: {citation['year']}-{citation['month']:02d}\n"
            f"Content:\n{hit['chunk_text'][:800]}"
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
