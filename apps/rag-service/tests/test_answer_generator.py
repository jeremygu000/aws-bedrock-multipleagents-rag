from __future__ import annotations

from app.answer_generator import (
    BedrockConverseAnswerGenerator,
    QwenAnswerGenerator,
    RoutedAnswerGenerator,
    _build_context_block,
    _extract_bedrock_text,
)
from app.config import Settings


def _sample_hits() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "chunk_text": "This is evidence snippet.",
            "score": 0.2,
            "citation": {
                "title": "Doc A",
                "url": "https://example.com/a",
                "year": 2025,
                "month": 4,
            },
        }
    ]


def test_build_context_block_contains_citation_fields() -> None:
    block = _build_context_block(_sample_hits())
    assert "[1] title: Doc A" in block
    assert "[1] url: https://example.com/a" in block
    assert "[1] date: 2025-04" in block


def test_extract_bedrock_text_happy_path() -> None:
    payload = {"output": {"message": {"content": [{"text": "line1"}, {"text": "line2"}]}}}
    assert _extract_bedrock_text(payload) == "line1\nline2"


def test_bedrock_generator_no_hits_returns_default_message() -> None:
    generator = BedrockConverseAnswerGenerator(Settings())
    answer = generator.generate("query", [])
    assert "could not find grounded passages" in answer.lower()


def test_bedrock_generator_uses_client(monkeypatch) -> None:
    generator = BedrockConverseAnswerGenerator(Settings(RAG_AWS_REGION="ap-southeast-2"))

    class FakeClient:
        def converse(self, **kwargs):
            assert kwargs["modelId"] == "amazon.nova-lite-v1:0"
            return {"output": {"message": {"content": [{"text": "bedrock-answer"}]}}}

    monkeypatch.setattr(generator, "_get_client", lambda: FakeClient())
    answer = generator.generate("query", _sample_hits())
    assert answer == "bedrock-answer"


def test_routed_generator_prefers_qwen_when_available() -> None:
    class FakeBedrock:
        def generate(self, query, hits):
            return "bedrock"

    class FakeQwen:
        def is_available(self):
            return True

        def generate(self, query, hits):
            return "qwen"

    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    answer, model = router.generate("q", _sample_hits(), "qwen-plus")
    assert answer == "qwen"
    assert model == "qwen-plus"


def test_routed_generator_fallback_qwen_to_bedrock() -> None:
    class FakeBedrock:
        def generate(self, query, hits):
            return "bedrock-fallback"

    class FakeQwen:
        def is_available(self):
            return True

        def generate(self, query, hits):
            raise RuntimeError("qwen failed")

    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    answer, model = router.generate("q", _sample_hits(), "qwen-plus")
    assert answer == "bedrock-fallback"
    assert model == "nova-lite"


def test_qwen_answer_generator_returns_default_when_no_hits() -> None:
    class FakeQwenClient:
        def is_configured(self):
            return True

        def chat(self, system_prompt: str, user_prompt: str) -> str:
            return "unused"

    generator = QwenAnswerGenerator(FakeQwenClient())
    answer = generator.generate("q", [])
    assert "could not find grounded passages" in answer.lower()


def test_qwen_answer_generator_calls_client() -> None:
    class FakeQwenClient:
        def is_configured(self):
            return True

        def chat(self, system_prompt: str, user_prompt: str) -> str:
            assert "Evidence" in user_prompt
            return "qwen-answer"

    generator = QwenAnswerGenerator(FakeQwenClient())
    assert generator.generate("q", _sample_hits()) == "qwen-answer"
