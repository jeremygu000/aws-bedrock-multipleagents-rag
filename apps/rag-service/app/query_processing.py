"""Intent detection and query rewriting using Qwen with safe fallbacks."""

from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .qwen_client import QwenClient


class QueryProcessor:
    """Query pre-processing pipeline used before retrieval."""

    def __init__(self, settings: Settings, qwen_client: QwenClient) -> None:
        """Store settings and Qwen client used by processing steps."""

        self._settings = settings
        self._qwen = qwen_client

    def detect_intent(self, query: str) -> dict[str, Any]:
        """Detect intent and complexity for routing decisions.

        Returns a dictionary with:
        - `intent`: short label like `factual`, `analytical`, or `procedural`
        - `complexity`: `low` / `medium` / `high`
        """

        if self._qwen.is_configured():
            system_prompt = (
                "You are an intent classifier for enterprise RAG queries. "
                "Return valid JSON only."
            )
            user_prompt = (
                "Classify this query. Return JSON: "
                '{"intent":"factual|analytical|procedural|comparison|other",'
                '"complexity":"low|medium|high"}'
                f"\nQuery: {query}"
            )
            try:
                raw = self._qwen.chat(system_prompt, user_prompt)
                parsed = _safe_json(raw)
                intent = str(parsed.get("intent", "")).strip().lower()
                complexity = str(parsed.get("complexity", "")).strip().lower()
                if intent and complexity in {"low", "medium", "high"}:
                    return {"intent": intent, "complexity": complexity}
            except Exception:
                # Fall back to deterministic heuristics when LLM classification fails.
                pass

        return self._heuristic_intent(query)

    def rewrite_query(self, query: str, intent: str, complexity: str) -> str:
        """Rewrite query for retrieval quality while preserving user intent."""

        if not self._settings.enable_query_rewrite:
            return query
        if not self._qwen.is_configured():
            return query

        system_prompt = (
            "You are a retrieval query rewriter. "
            "Rewrite user query for keyword+dense retrieval. "
            "Preserve entities, years, and constraints. "
            "Return one line only."
        )
        user_prompt = (
            f"Intent: {intent}\n"
            f"Complexity: {complexity}\n"
            f"Original query: {query}\n"
            "Rewrite:"
        )
        try:
            rewritten = self._qwen.chat(system_prompt, user_prompt).strip()
            if rewritten:
                return rewritten
        except Exception:
            return query
        return query

    def _heuristic_intent(self, query: str) -> dict[str, Any]:
        """Fallback intent/complexity classifier based on simple lexical cues."""

        lowered = query.lower()
        complex_markers = [
            "compare",
            "difference",
            "trade-off",
            "why",
            "how",
            "步骤",
            "对比",
            "原因",
            "影响",
        ]
        token_count = len(query.split())
        is_complex = token_count >= self._settings.route_complex_query_token_threshold or any(
            marker in lowered for marker in complex_markers
        )
        if "compare" in lowered or "对比" in lowered:
            intent = "comparison"
        elif "how" in lowered or "步骤" in lowered:
            intent = "procedural"
        elif "why" in lowered or "原因" in lowered:
            intent = "analytical"
        else:
            intent = "factual"
        return {
            "intent": intent,
            "complexity": "high" if is_complex else "medium",
        }


def _safe_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction for model outputs that may include wrappers."""

    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        parsed = json.loads(stripped[first : last + 1])
        return parsed if isinstance(parsed, dict) else {}
    return {}
