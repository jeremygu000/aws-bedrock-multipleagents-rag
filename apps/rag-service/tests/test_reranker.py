from __future__ import annotations

from app.config import Settings
from app.reranker import LLMReranker


class FakeQwen:
    def __init__(self, configured: bool, responses: list[str] | None = None) -> None:
        self._configured = configured
        self._responses = responses or []
        self.calls: list[tuple[str, str]] = []

    def is_configured(self) -> bool:
        return self._configured

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise ValueError("no response queued")
        return self._responses.pop(0)


def _sample_hits(n: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": f"c{i}",
            "chunk_text": f"content for chunk {i}",
            "score": 0.5 - i * 0.05,
            "citation": {"title": f"Doc {i}", "url": f"https://example.com/{i}", "year": 2025, "month": 1},
        }
        for i in range(n)
    ]


def test_rerank_sorts_by_llm_score() -> None:
    scores_json = '[{"chunk_id":"c2","score":10},{"chunk_id":"c0","score":5},{"chunk_id":"c1","score":3}]'
    qwen = FakeQwen(configured=True, responses=[scores_json])
    reranker = LLMReranker(settings=Settings(), qwen_client=qwen)
    result = reranker.rerank("query", _sample_hits(3), top_k=3)
    assert [h["chunk_id"] for h in result] == ["c2", "c0", "c1"]


def test_rerank_returns_top_k() -> None:
    scores_json = '[{"chunk_id":"c0","score":10},{"chunk_id":"c1","score":8},{"chunk_id":"c2","score":6},{"chunk_id":"c3","score":4},{"chunk_id":"c4","score":2}]'
    qwen = FakeQwen(configured=True, responses=[scores_json])
    reranker = LLMReranker(settings=Settings(), qwen_client=qwen)
    result = reranker.rerank("query", _sample_hits(5), top_k=2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c0"


def test_rerank_fallback_on_error() -> None:
    qwen = FakeQwen(configured=True, responses=[])
    reranker = LLMReranker(settings=Settings(), qwen_client=qwen)
    hits = _sample_hits(3)
    result = reranker.rerank("query", hits, top_k=2)
    assert len(result) == 2
    assert result[0]["chunk_id"] == "c0"


def test_rerank_passthrough_when_disabled() -> None:
    qwen = FakeQwen(configured=True, responses=["should-not-be-used"])
    reranker = LLMReranker(settings=Settings(RAG_ENABLE_RERANKING="false"), qwen_client=qwen)
    hits = _sample_hits(5)
    result = reranker.rerank("query", hits, top_k=3)
    assert len(result) == 3
    assert len(qwen.calls) == 0


def test_rerank_empty_hits() -> None:
    qwen = FakeQwen(configured=True)
    reranker = LLMReranker(settings=Settings(), qwen_client=qwen)
    result = reranker.rerank("query", [], top_k=5)
    assert result == []
