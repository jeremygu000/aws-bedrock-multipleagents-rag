"""CRAG (Corrective RAG) components: retrieval grading, query rewriting, web search."""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import Settings
from .qwen_client import QwenClient

logger = logging.getLogger(__name__)

_GRADER_SYSTEM_PROMPT = (
    "You are a relevance grader. Given a user question and a retrieved document, "
    "determine whether the document contains information relevant to the question. "
    "Respond with ONLY a JSON object: {\"relevant\": true} or {\"relevant\": false}. "
    "A document is relevant if it contains keywords or semantic meaning related to the question. "
    "It does not need to fully answer the question."
)

_REWRITER_SYSTEM_PROMPT = (
    "You are a query rewriter. The user's original query did not find good results "
    "in the internal knowledge base. Rewrite the query for web search. "
    "Requirements: preserve the core intent, add specific keywords and context, "
    "expand abbreviations and jargon, make the query more specific and searchable. "
    "Return ONLY the rewritten query text, nothing else."
)


class RetrievalGrader:
    def __init__(self, settings: Settings, qwen_client: QwenClient) -> None:
        self._settings = settings
        self._qwen = qwen_client

    def grade(self, question: str, document_text: str) -> bool:
        if not self._qwen.is_configured():
            return True

        user_prompt = f"Question: {question}\n\nDocument:\n{document_text[:2000]}"
        try:
            raw = self._qwen.chat(_GRADER_SYSTEM_PROMPT, user_prompt, max_tokens=50)
            return self._parse_relevance(raw)
        except Exception:
            logger.exception("Retrieval grading failed, assuming relevant")
            return True

    def grade_hits(
        self,
        question: str,
        hits: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        if not hits:
            return [], "incorrect"

        relevant: list[dict[str, Any]] = []
        for hit in hits:
            if self.grade(question, hit.get("chunk_text", "")):
                relevant.append(hit)

        ratio = len(relevant) / len(hits)
        upper = self._settings.crag_upper_threshold
        lower = self._settings.crag_lower_threshold
        min_docs = self._settings.crag_min_relevant_docs

        if len(relevant) >= min_docs and ratio >= upper:
            verdict = "correct"
        elif ratio <= lower:
            verdict = "incorrect"
        else:
            verdict = "ambiguous"

        return relevant, verdict

    @staticmethod
    def _parse_relevance(raw: str) -> bool:
        stripped = raw.strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return "true" in stripped.lower()
        try:
            parsed = json.loads(stripped[start : end + 1])
            return bool(parsed.get("relevant", False))
        except (json.JSONDecodeError, ValueError):
            return "true" in stripped.lower()


class CragQueryRewriter:
    def __init__(self, settings: Settings, qwen_client: QwenClient) -> None:
        self._settings = settings
        self._qwen = qwen_client

    def rewrite(self, query: str) -> str:
        if not self._qwen.is_configured():
            return query

        try:
            rewritten = self._qwen.chat(_REWRITER_SYSTEM_PROMPT, query, max_tokens=200)
            return rewritten.strip() or query
        except Exception:
            logger.exception("CRAG query rewrite failed, using original query")
            return query


class CragWebSearcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def search(self, query: str) -> list[dict[str, Any]]:
        if not self._settings.crag_enable_web_search:
            return []

        api_key = self._settings.tavily_api_key
        if not api_key:
            logger.warning("CRAG web search enabled but TAVILY_API_KEY is empty")
            return []

        try:
            return self._tavily_search(query, api_key)
        except Exception:
            logger.exception("CRAG web search failed")
            return []

    def _tavily_search(self, query: str, api_key: str) -> list[dict[str, Any]]:
        import http.client
        from urllib import error, request

        url = "https://api.tavily.com/search"
        payload = {
            "query": query,
            "max_results": self._settings.crag_web_search_k,
            "search_depth": "basic",
            "include_answer": False,
        }
        body = json.dumps({"api_key": api_key, **payload}).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except (error.HTTPError, error.URLError, http.client.HTTPException) as exc:
            logger.warning("Tavily API error: %s", exc)
            return []

        parsed = json.loads(raw)
        results: list[dict[str, Any]] = []
        for item in parsed.get("results", []):
            results.append(
                {
                    "chunk_id": f"web_{hash(item.get('url', '')) & 0xFFFFFFFF:08x}",
                    "chunk_text": item.get("content", ""),
                    "score": item.get("score", 0.5),
                    "citation": {
                        "title": item.get("title", "Web Result"),
                        "url": item.get("url", ""),
                        "year": 2026,
                        "month": 1,
                    },
                    "source": "web_search",
                }
            )
        return results
