"""Answer generation strategies and model routing for RAG responses."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Literal

import boto3

from .config import Settings
from .models import GraphContext
from .qwen_client import QwenClient
from .tracing import LLM_TOKEN_USAGE

logger = logging.getLogger(__name__)

ModelRoute = Literal["nova-lite", "qwen-plus"]


def _short_model_label(model_id: str) -> str:
    parts = model_id.rsplit("/", 1)
    name = parts[-1] if parts else model_id
    for prefix in ("au.", "us.", "eu.", "global."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.split(":")[0] if ":" in name else name

_GENERIC_SYSTEM_PROMPT = (
    "You are a grounded enterprise RAG assistant. "
    "Answer directly using the provided evidence. "
    "IMPORTANT: Quote specific facts, names, numbers, and dates from the evidence verbatim. "
    "Do not paraphrase or generalize when the evidence contains exact information. "
    "Use citation markers [1], [2], ... to reference evidence sources. "
    "Structure your response: first state the key answer with specific details, then provide supporting context with citations. "
    "Only state 'I could not find information about [topic]' if the evidence is genuinely irrelevant to the question."
)

_INTENT_SYSTEM_PROMPTS: dict[str, str] = {
    "factual": (
        "You are a grounded enterprise RAG assistant. "
        "Answer ONLY using the provided evidence. "
        "Be precise: quote exact names, numbers, dates, and terminology from the evidence. "
        "Never generalize when specific facts are available. "
        "Cite every factual claim with [N] markers. "
        "If the evidence doesn't contain the answer, say so."
    ),
    "analytical": (
        "You are a grounded enterprise RAG assistant. "
        "Analyze the evidence to answer this question. "
        "Include specific data points, names, and figures from the evidence. "
        "Structure your response: "
        "1) Key finding with specific details, "
        "2) Supporting analysis with citations [N] and quoted evidence, "
        "3) Caveats or limitations based on available evidence."
    ),
    "procedural": (
        "You are a grounded enterprise RAG assistant. "
        "Provide step-by-step instructions based on the evidence. "
        "Include specific names, URLs, and requirements mentioned in the evidence. "
        "Number each step clearly. "
        "Cite the source for each step with [N] markers."
    ),
    "comparison": (
        "You are a grounded enterprise RAG assistant. "
        "Compare the items using evidence provided. "
        "Use specific facts, figures, and terminology from the evidence — do not paraphrase. "
        "Structure as: "
        "1) Summary of key differences with specific details, "
        "2) Detailed comparison with citations [N], "
        "3) Recommendation if evidence supports one."
    ),
}


_KG_REASONING_SUPPLEMENT = (
    "\n\nYou also have access to a Knowledge Graph with entity and relationship data. "
    "When entity relationships are relevant, cite them explicitly: "
    '"According to the knowledge graph, [Entity A] [relation] [Entity B]..." '
    "If the answer requires connecting multiple entities, explain the reasoning path. "
    "Distinguish between graph-inferred context and text-grounded facts."
)


def _get_intent_system_prompt(
    intent: str, settings: Settings, *, has_graph_context: bool = False
) -> str:
    """Return system prompt based on intent and feature flag.

    Args:
        intent: Detected query intent (factual, analytical, procedural, comparison, other).
        settings: Runtime settings controlling feature flags.
        has_graph_context: When True, append KG reasoning instructions.

    Returns:
        System prompt string appropriate for the intent.
    """
    if not settings.enable_intent_aware_prompts:
        base = _GENERIC_SYSTEM_PROMPT
    else:
        base = _INTENT_SYSTEM_PROMPTS.get(intent, _GENERIC_SYSTEM_PROMPT)

    if has_graph_context:
        return base + _KG_REASONING_SUPPLEMENT
    return base


def _build_graph_evidence_block(graph_context: GraphContext | None) -> str:
    """Build a formatted graph evidence section for injection into answer prompts.

    Args:
        graph_context: Graph context with entities and relations, or None.

    Returns:
        Formatted string with entity/relation sections, or empty string.
    """
    if graph_context is None or graph_context.is_empty:
        return ""
    return graph_context.to_evidence_text()


def _build_community_evidence_block(
    community_summaries: list[dict[str, Any]] | None,
) -> str:
    if not community_summaries:
        return ""
    parts = []
    for i, cs in enumerate(community_summaries, 1):
        findings_text = "; ".join(cs.get("findings", []))
        parts.append(
            f"--- Community [{i}] ---\n"
            f"Topic: {cs.get('title', 'Unknown')}\n"
            f"Summary: {cs.get('summary', '')}\n"
            f"Key Findings: {findings_text}\n"
            f"Entities: {cs.get('entity_count', 0)} | Rating: {cs.get('rating', 0):.1f}/10"
        )
    return "\n\n".join(parts)


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
        graph_context: GraphContext | None = None,
        community_summaries: list[dict[str, Any]] | None = None,
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
        graph_evidence = _build_graph_evidence_block(graph_context)
        community_evidence = _build_community_evidence_block(community_summaries)
        has_graph = bool(graph_evidence) or bool(community_evidence)
        system_prompt = _get_intent_system_prompt(
            intent, self._settings, has_graph_context=has_graph
        )
        keyword_line = ""
        if keywords:
            keyword_line = f"Key topics: {', '.join(keywords)}\n"

        evidence_parts: list[str] = []
        if graph_evidence:
            evidence_parts.append(f"=== KNOWLEDGE GRAPH ===\n{graph_evidence}")
        if community_evidence:
            evidence_parts.append(f"=== COMMUNITY CONTEXT ===\n{community_evidence}")
        evidence_parts.append(f"=== TEXT EVIDENCE ===\n{context_block}")
        evidence_section = "\n\n".join(evidence_parts)

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"{keyword_line}"
            f"Evidence:\n{evidence_section}\n\n"
            "Write a detailed answer quoting specific facts, names, numbers, and dates from the evidence. "
            "Use [N] citation markers for every claim."
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
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        model_label = _short_model_label(self._settings.answer_model_id)
        if input_tokens:
            LLM_TOKEN_USAGE.labels(model=model_label, direction="input").inc(input_tokens)
        if output_tokens:
            LLM_TOKEN_USAGE.labels(model=model_label, direction="output").inc(output_tokens)
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

    def generate_stream(
        self,
        query: str,
        hits: list[dict[str, Any]],
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
        graph_context: GraphContext | None = None,
        community_summaries: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Stream answer tokens using Bedrock converse_stream API."""

        if not hits:
            yield "I could not find grounded passages for this query in the current knowledge base."
            return

        context_block = _build_context_block(
            hits,
            max_chars=self._settings.answer_evidence_max_chars,
            include_scores=self._settings.enable_relevance_scores_in_evidence,
        )
        graph_evidence = _build_graph_evidence_block(graph_context)
        community_evidence = _build_community_evidence_block(community_summaries)
        has_graph = bool(graph_evidence) or bool(community_evidence)
        system_prompt = _get_intent_system_prompt(
            intent, self._settings, has_graph_context=has_graph
        )
        keyword_line = ""
        if keywords:
            keyword_line = f"Key topics: {', '.join(keywords)}\n"

        evidence_parts: list[str] = []
        if graph_evidence:
            evidence_parts.append(f"=== KNOWLEDGE GRAPH ===\n{graph_evidence}")
        if community_evidence:
            evidence_parts.append(f"=== COMMUNITY CONTEXT ===\n{community_evidence}")
        evidence_parts.append(f"=== TEXT EVIDENCE ===\n{context_block}")
        evidence_section = "\n\n".join(evidence_parts)

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"{keyword_line}"
            f"Evidence:\n{evidence_section}\n\n"
            "Write a detailed answer quoting specific facts, names, numbers, and dates from the evidence. "
            "Use [N] citation markers for every claim."
        )

        client = self._get_client()
        response = client.converse_stream(
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
        for event in response.get("stream", []):
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            text = delta.get("text")
            if text:
                yield text


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
        graph_context: GraphContext | None = None,
        community_summaries: list[dict[str, Any]] | None = None,
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
        graph_evidence = _build_graph_evidence_block(graph_context)
        community_evidence = _build_community_evidence_block(community_summaries)
        has_graph = bool(graph_evidence) or bool(community_evidence)
        system_prompt = _get_intent_system_prompt(
            intent, self._settings, has_graph_context=has_graph
        )
        keyword_line = ""
        if keywords:
            keyword_line = f"Key topics: {', '.join(keywords)}\n"

        evidence_parts: list[str] = []
        if graph_evidence:
            evidence_parts.append(f"=== KNOWLEDGE GRAPH ===\n{graph_evidence}")
        if community_evidence:
            evidence_parts.append(f"=== COMMUNITY CONTEXT ===\n{community_evidence}")
        evidence_parts.append(f"=== TEXT EVIDENCE ===\n{context_block}")
        evidence_section = "\n\n".join(evidence_parts)

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"{keyword_line}"
            f"Evidence:\n{evidence_section}\n\n"
            "Write a detailed answer quoting specific facts, names, numbers, and dates from the evidence. "
            "Use [N] citation markers for every claim."
        )
        answer = self._qwen.chat(system_prompt, user_prompt).strip()
        input_tokens = int(len(user_prompt.split()) * 1.3)
        output_tokens = int(len(answer.split()) * 1.3)
        if input_tokens:
            LLM_TOKEN_USAGE.labels(model="qwen-plus", direction="input").inc(input_tokens)
        if output_tokens:
            LLM_TOKEN_USAGE.labels(model="qwen-plus", direction="output").inc(output_tokens)
        if not answer:
            return "I could not produce a grounded answer from the current evidence."
        return answer

    def generate_stream(
        self,
        query: str,
        hits: list[dict[str, Any]],
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
        graph_context: GraphContext | None = None,
        community_summaries: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        """Stream answer tokens using Qwen chat-completions streaming."""

        if not hits:
            yield "I could not find grounded passages for this query in the current knowledge base."
            return

        context_block = _build_context_block(
            hits,
            max_chars=self._settings.answer_evidence_max_chars,
            include_scores=self._settings.enable_relevance_scores_in_evidence,
        )
        graph_evidence = _build_graph_evidence_block(graph_context)
        community_evidence = _build_community_evidence_block(community_summaries)
        has_graph = bool(graph_evidence) or bool(community_evidence)
        system_prompt = _get_intent_system_prompt(
            intent, self._settings, has_graph_context=has_graph
        )
        keyword_line = ""
        if keywords:
            keyword_line = f"Key topics: {', '.join(keywords)}\n"

        evidence_parts: list[str] = []
        if graph_evidence:
            evidence_parts.append(f"=== KNOWLEDGE GRAPH ===\n{graph_evidence}")
        if community_evidence:
            evidence_parts.append(f"=== COMMUNITY CONTEXT ===\n{community_evidence}")
        evidence_parts.append(f"=== TEXT EVIDENCE ===\n{context_block}")
        evidence_section = "\n\n".join(evidence_parts)

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"{keyword_line}"
            f"Evidence:\n{evidence_section}\n\n"
            "Write a detailed answer quoting specific facts, names, numbers, and dates from the evidence. "
            "Use [N] citation markers for every claim."
        )
        yield from self._qwen.stream_chat(system_prompt, user_prompt)


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
        graph_context: GraphContext | None = None,
        community_summaries: list[dict[str, Any]] | None = None,
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
                        query,
                        hits,
                        intent=intent,
                        complexity=complexity,
                        keywords=keywords,
                        graph_context=graph_context,
                        community_summaries=community_summaries,
                    ),
                    "qwen-plus",
                )
            except Exception:
                return (
                    self._bedrock.generate(
                        query,
                        hits,
                        intent=intent,
                        complexity=complexity,
                        keywords=keywords,
                        graph_context=graph_context,
                        community_summaries=community_summaries,
                    ),
                    "nova-lite",
                )

        if preferred_model == "nova-lite":
            try:
                return (
                    self._bedrock.generate(
                        query,
                        hits,
                        intent=intent,
                        complexity=complexity,
                        keywords=keywords,
                        graph_context=graph_context,
                        community_summaries=community_summaries,
                    ),
                    "nova-lite",
                )
            except Exception:
                if self._qwen.is_available():
                    return (
                        self._qwen.generate(
                            query,
                            hits,
                            intent=intent,
                            complexity=complexity,
                            keywords=keywords,
                            graph_context=graph_context,
                            community_summaries=community_summaries,
                        ),
                        "qwen-plus",
                    )
                raise

        return (
            self._bedrock.generate(
                query,
                hits,
                intent=intent,
                complexity=complexity,
                keywords=keywords,
                graph_context=graph_context,
                community_summaries=community_summaries,
            ),
            "nova-lite",
        )

    def generate_stream(
        self,
        query: str,
        hits: list[dict[str, Any]],
        preferred_model: ModelRoute,
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
        graph_context: GraphContext | None = None,
        community_summaries: list[dict[str, Any]] | None = None,
    ) -> tuple[Iterator[str], ModelRoute]:
        """Stream answer using preferred model with automatic fallback.

        Returns an iterator of text chunks and the actual model used.
        Unlike generate(), fallback only applies at connection time — once
        streaming starts, the chosen backend is committed.
        """

        gen_kwargs: dict[str, Any] = {
            "query": query,
            "hits": hits,
            "intent": intent,
            "complexity": complexity,
            "keywords": keywords,
            "graph_context": graph_context,
            "community_summaries": community_summaries,
        }

        if preferred_model == "qwen-plus" and self._qwen.is_available():
            try:
                stream = self._qwen.generate_stream(**gen_kwargs)
                first_chunk = next(stream)

                def _qwen_stream() -> Iterator[str]:
                    yield first_chunk
                    yield from stream

                return _qwen_stream(), "qwen-plus"
            except Exception:
                logger.warning("Qwen stream failed, falling back to Bedrock")
                return self._bedrock.generate_stream(**gen_kwargs), "nova-lite"

        if preferred_model == "nova-lite":
            try:
                stream = self._bedrock.generate_stream(**gen_kwargs)
                first_chunk = next(stream)

                def _bedrock_stream() -> Iterator[str]:
                    yield first_chunk
                    yield from stream

                return _bedrock_stream(), "nova-lite"
            except Exception:
                if self._qwen.is_available():
                    logger.warning("Bedrock stream failed, falling back to Qwen")
                    return self._qwen.generate_stream(**gen_kwargs), "qwen-plus"
                raise

        return self._bedrock.generate_stream(**gen_kwargs), "nova-lite"


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
