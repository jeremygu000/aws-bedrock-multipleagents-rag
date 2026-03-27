from __future__ import annotations

from app.config import Settings
from app.query_processing import QueryProcessor, _safe_json


class FakeQwen:
    def __init__(
        self,
        configured: bool,
        responses: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self._configured = configured
        self._responses = responses or []
        self._embeddings = embeddings or []
        self.calls: list[tuple[str, str]] = []

    def is_configured(self) -> bool:
        return self._configured

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise ValueError("no response queued")
        return self._responses.pop(0)

    def embedding(self, text: str) -> list[float]:
        if not self._embeddings:
            raise ValueError("no embedding queued")
        return self._embeddings.pop(0)


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


def test_build_query_embedding_returns_vector_when_valid() -> None:
    qwen = FakeQwen(configured=True, embeddings=[[0.1, 0.2, 0.3]])
    qp = QueryProcessor(settings=Settings(RAG_EMBED_DIM="3"), qwen_client=qwen)
    assert qp.build_query_embedding("query") == [0.1, 0.2, 0.3]


def test_build_query_embedding_returns_none_on_dimension_mismatch() -> None:
    qwen = FakeQwen(configured=True, embeddings=[[0.1, 0.2]])
    qp = QueryProcessor(settings=Settings(RAG_EMBED_DIM="3"), qwen_client=qwen)
    assert qp.build_query_embedding("query") is None


def test_extract_keywords_returns_parsed_result() -> None:
    qwen = FakeQwen(
        configured=True,
        responses=['{"hl_keywords":["market trends"],"ll_keywords":["ACME Corp","Q3 2024"]}'],
    )
    qp = QueryProcessor(
        settings=Settings(RAG_ENABLE_KEYWORD_EXTRACTION="true"),
        qwen_client=qwen,
    )
    result = qp.extract_keywords("What are ACME Corp market trends in Q3 2024?")
    assert result.hl_keywords == ["market trends"]
    assert result.ll_keywords == ["ACME Corp", "Q3 2024"]


def test_extract_keywords_returns_empty_when_disabled() -> None:
    qwen = FakeQwen(configured=True, responses=["should-not-be-used"])
    qp = QueryProcessor(
        settings=Settings(RAG_ENABLE_KEYWORD_EXTRACTION="false"),
        qwen_client=qwen,
    )
    result = qp.extract_keywords("any query")
    assert result.hl_keywords == []
    assert result.ll_keywords == []
    assert len(qwen.calls) == 0


def test_extract_keywords_fallback_on_malformed_json() -> None:
    qwen = FakeQwen(configured=True, responses=["not valid json"])
    qp = QueryProcessor(
        settings=Settings(RAG_ENABLE_KEYWORD_EXTRACTION="true"),
        qwen_client=qwen,
    )
    result = qp.extract_keywords("any query")
    assert result.hl_keywords == []
    assert result.ll_keywords == []


def test_rewrite_query_includes_keywords() -> None:
    qwen = FakeQwen(configured=True, responses=["expanded rewrite"])
    qp = QueryProcessor(
        settings=Settings(RAG_ENABLE_QUERY_REWRITE="true"),
        qwen_client=qwen,
    )
    rewritten = qp.rewrite_query("original", "factual", "medium", ll_keywords=["ACME", "Q3"])
    assert rewritten == "expanded rewrite"
    assert "ACME" in qwen.calls[0][1]
    assert "Q3" in qwen.calls[0][1]

