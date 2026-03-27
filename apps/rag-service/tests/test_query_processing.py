from __future__ import annotations

from app.config import Settings
from app.query_processing import QueryProcessor, _safe_json


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


def test_detect_intent_from_qwen_json() -> None:
    qp = QueryProcessor(
        settings=Settings(),
        qwen_client=FakeQwen(
            configured=True,
            responses=['{"intent":"analytical","complexity":"high"}'],
        ),
    )
    result = qp.detect_intent("Why did policy change?")
    assert result == {"intent": "analytical", "complexity": "high"}


def test_detect_intent_fallback_heuristic() -> None:
    qp = QueryProcessor(
        settings=Settings(),
        qwen_client=FakeQwen(configured=False),
    )
    result = qp.detect_intent("How do I complete the registration steps?")
    assert result["intent"] == "procedural"
    assert result["complexity"] in {"medium", "high"}


def test_rewrite_query_uses_qwen_when_enabled() -> None:
    qwen = FakeQwen(configured=True, responses=["rewritten query"])
    qp = QueryProcessor(settings=Settings(RAG_ENABLE_QUERY_REWRITE="true"), qwen_client=qwen)
    rewritten = qp.rewrite_query("original", "factual", "medium")
    assert rewritten == "rewritten query"
    assert len(qwen.calls) == 1


def test_rewrite_query_returns_original_when_disabled() -> None:
    qwen = FakeQwen(configured=True, responses=["should-not-be-used"])
    qp = QueryProcessor(settings=Settings(RAG_ENABLE_QUERY_REWRITE="false"), qwen_client=qwen)
    assert qp.rewrite_query("original", "factual", "medium") == "original"
    assert len(qwen.calls) == 0


def test_safe_json_extracts_wrapped_json() -> None:
    raw = 'some prefix {"intent":"factual","complexity":"medium"} some suffix'
    assert _safe_json(raw) == {"intent": "factual", "complexity": "medium"}
