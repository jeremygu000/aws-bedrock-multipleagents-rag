"""LLM-based reranking of retrieval results inspired by LightRAG."""

from __future__ import annotations

import json
from typing import Any

from .config import Settings
from .qwen_client import QwenClient


class LLMReranker:
    """Rerank retrieval hits using Qwen Plus LLM scoring."""

    def __init__(self, settings: Settings, qwen_client: QwenClient) -> None:
        self._settings = settings
        self._qwen = qwen_client

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Rerank hits by LLM relevance score, returning top_k results.

        Falls back to original hits (truncated to top_k) on any failure.
        """
        if not self._settings.enable_reranking:
            return hits[:top_k]
        if not hits:
            return []
        if not self._qwen.is_configured():
            return hits[:top_k]

        candidates = hits[: self._settings.rerank_candidate_count]
        evidence_lines = self._build_evidence(candidates)

        system_prompt = (
            "You are a relevance scoring engine. "
            "Score each evidence chunk 0-10 for relevance to the query. "
            "Return valid JSON only: a list of objects with chunk_id and score."
        )
        user_prompt = (
            f"Query: {query}\n\n"
            f"Evidence chunks:\n{evidence_lines}\n\n"
            "Score each chunk 0-10. Return JSON array: "
            '[{"chunk_id": "...", "score": N}, ...]'
        )

        try:
            raw = self._qwen.chat(system_prompt, user_prompt)
            scores = self._parse_scores(raw, candidates)
            scored = sorted(
                candidates,
                key=lambda h: scores.get(h["chunk_id"], 0.0),
                reverse=True,
            )
            return scored[:top_k]
        except Exception:
            return hits[:top_k]

    def _build_evidence(self, candidates: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        char_budget = self._settings.rerank_max_tokens * 4
        total_chars = 0
        for hit in candidates:
            chunk_text = hit.get("chunk_text", "")
            max_chunk_chars = char_budget // max(len(candidates), 1)
            truncated = chunk_text[:max_chunk_chars]
            line = f"[{hit['chunk_id']}] {truncated}"
            if total_chars + len(line) > char_budget:
                break
            lines.append(line)
            total_chars += len(line)
        return "\n\n".join(lines)

    def _parse_scores(self, raw: str, candidates: list[dict[str, Any]]) -> dict[str, float]:
        stripped = raw.strip()
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start < 0 or end <= start:
            return {}
        parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, list):
            return {}
        scores: dict[str, float] = {}
        valid_ids = {h["chunk_id"] for h in candidates}
        for item in parsed:
            if isinstance(item, dict):
                cid = str(item.get("chunk_id", ""))
                score = item.get("score", 0)
                if cid in valid_ids and isinstance(score, (int, float)):
                    scores[cid] = float(score)
        return scores
