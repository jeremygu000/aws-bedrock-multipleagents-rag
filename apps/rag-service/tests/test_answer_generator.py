from __future__ import annotations

from app.answer_generator import (
    _KG_REASONING_SUPPLEMENT,
    BedrockConverseAnswerGenerator,
    QwenAnswerGenerator,
    RoutedAnswerGenerator,
    _build_context_block,
    _build_graph_evidence_block,
    _extract_bedrock_text,
    _get_intent_system_prompt,
)
from app.config import Settings
from app.models import GraphContext, GraphEntity, GraphRelation


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
    assert "Title: Doc A" in block
    assert "URL: https://example.com/a" in block
    assert "Date: 2025-04" in block


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
            assert kwargs["modelId"] == "amazon.nova-pro-v1:0"
            return {"output": {"message": {"content": [{"text": "bedrock-answer"}]}}}

    monkeypatch.setattr(generator, "_get_client", lambda: FakeClient())
    answer = generator.generate("query", _sample_hits())
    assert answer == "bedrock-answer"


def test_routed_generator_prefers_qwen_when_available() -> None:
    class FakeBedrock:
        def generate(self, query, hits, **kwargs):
            return "bedrock"

    class FakeQwen:
        def is_available(self):
            return True

        def generate(self, query, hits, **kwargs):
            return "qwen"

    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    answer, model = router.generate("q", _sample_hits(), "qwen-plus")
    assert answer == "qwen"
    assert model == "qwen-plus"


def test_routed_generator_fallback_qwen_to_bedrock() -> None:
    class FakeBedrock:
        def generate(self, query, hits, **kwargs):
            return "bedrock-fallback"

    class FakeQwen:
        def is_available(self):
            return True

        def generate(self, query, hits, **kwargs):
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

    generator = QwenAnswerGenerator(FakeQwenClient(), Settings())
    answer = generator.generate("q", [])
    assert "could not find grounded passages" in answer.lower()


def test_qwen_answer_generator_calls_client() -> None:
    class FakeQwenClient:
        def is_configured(self):
            return True

        def chat(self, system_prompt: str, user_prompt: str) -> str:
            assert "Evidence" in user_prompt
            return "qwen-answer"

    generator = QwenAnswerGenerator(FakeQwenClient(), Settings())
    assert generator.generate("q", _sample_hits()) == "qwen-answer"


# ---------------------------------------------------------------------------
# Phase 3.4 — Graph context injection tests
# ---------------------------------------------------------------------------


def _sample_graph_context() -> GraphContext:
    return GraphContext(
        entities=[
            GraphEntity(
                entity_id="e1",
                name="ACME Corp",
                type="Organization",
                description="Leading AI company",
                score=0.9,
            ),
            GraphEntity(
                entity_id="e2",
                name="John Smith",
                type="Person",
                description="CTO of ACME Corp",
                score=0.8,
            ),
        ],
        relations=[
            GraphRelation(
                source_entity="John Smith",
                target_entity="ACME Corp",
                relation_type="works_at",
                evidence="Appointed CTO in 2018",
                score=0.85,
            ),
        ],
        source_chunk_ids=["c1"],
    )


def test_build_graph_evidence_block_none() -> None:
    assert _build_graph_evidence_block(None) == ""


def test_build_graph_evidence_block_empty() -> None:
    assert _build_graph_evidence_block(GraphContext()) == ""


def test_build_graph_evidence_block_with_entities_and_relations() -> None:
    block = _build_graph_evidence_block(_sample_graph_context())
    assert "ACME Corp" in block
    assert "John Smith" in block
    assert "works_at" in block
    assert "Appointed CTO in 2018" in block


def test_system_prompt_includes_kg_supplement_when_graph_present() -> None:
    prompt = _get_intent_system_prompt("factual", Settings(), has_graph_context=True)
    assert "Knowledge Graph" in prompt
    assert _KG_REASONING_SUPPLEMENT in prompt


def test_system_prompt_no_kg_supplement_without_graph() -> None:
    prompt = _get_intent_system_prompt("factual", Settings(), has_graph_context=False)
    assert _KG_REASONING_SUPPLEMENT not in prompt


def test_bedrock_generator_injects_graph_context(monkeypatch) -> None:
    generator = BedrockConverseAnswerGenerator(Settings(RAG_AWS_REGION="ap-southeast-2"))
    captured_prompts: list[str] = []

    class FakeClient:
        def converse(self, **kwargs):
            user_msg = kwargs["messages"][0]["content"][0]["text"]
            captured_prompts.append(user_msg)
            system_text = kwargs["system"][0]["text"]
            captured_prompts.append(system_text)
            return {"output": {"message": {"content": [{"text": "answer-with-graph"}]}}}

    monkeypatch.setattr(generator, "_get_client", lambda: FakeClient())
    answer = generator.generate("query", _sample_hits(), graph_context=_sample_graph_context())
    assert answer == "answer-with-graph"
    user_prompt = captured_prompts[0]
    system_prompt = captured_prompts[1]
    assert "=== KNOWLEDGE GRAPH ===" in user_prompt
    assert "=== TEXT EVIDENCE ===" in user_prompt
    assert "ACME Corp" in user_prompt
    assert "Knowledge Graph" in system_prompt


def test_bedrock_generator_no_graph_context_no_kg_section(monkeypatch) -> None:
    generator = BedrockConverseAnswerGenerator(Settings(RAG_AWS_REGION="ap-southeast-2"))
    captured_prompts: list[str] = []

    class FakeClient:
        def converse(self, **kwargs):
            user_msg = kwargs["messages"][0]["content"][0]["text"]
            captured_prompts.append(user_msg)
            return {"output": {"message": {"content": [{"text": "answer-no-graph"}]}}}

    monkeypatch.setattr(generator, "_get_client", lambda: FakeClient())
    answer = generator.generate("query", _sample_hits(), graph_context=None)
    assert answer == "answer-no-graph"
    assert "=== KNOWLEDGE GRAPH ===" not in captured_prompts[0]
    assert "=== TEXT EVIDENCE ===" in captured_prompts[0]


def test_qwen_generator_injects_graph_context() -> None:
    captured: list[str] = []

    class FakeQwenClient:
        def is_configured(self):
            return True

        def chat(self, system_prompt: str, user_prompt: str) -> str:
            captured.append(user_prompt)
            captured.append(system_prompt)
            return "qwen-graph-answer"

    generator = QwenAnswerGenerator(FakeQwenClient(), Settings())
    answer = generator.generate("q", _sample_hits(), graph_context=_sample_graph_context())
    assert answer == "qwen-graph-answer"
    assert "=== KNOWLEDGE GRAPH ===" in captured[0]
    assert "ACME Corp" in captured[0]
    assert "Knowledge Graph" in captured[1]


def test_routed_generator_forwards_graph_context() -> None:
    received_kwargs: list[dict] = []

    class FakeBedrock:
        def generate(self, query, hits, **kwargs):
            received_kwargs.append(kwargs)
            return "bedrock"

    class FakeQwen:
        def is_available(self):
            return False

        def generate(self, query, hits, **kwargs):
            return "qwen"

    ctx = _sample_graph_context()
    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    answer, model = router.generate("q", _sample_hits(), "nova-lite", graph_context=ctx)
    assert answer == "bedrock"
    assert received_kwargs[0]["graph_context"] is ctx


def test_build_graph_evidence_block_entities_only() -> None:
    ctx = GraphContext(
        entities=[
            GraphEntity(
                entity_id="e1",
                name="TestOrg",
                type="Organization",
                description="A test org",
                score=0.5,
            ),
        ],
    )
    block = _build_graph_evidence_block(ctx)
    assert "TestOrg" in block
    assert "### Entities" in block
    assert "### Relations" not in block


# ---------------------------------------------------------------------------
# Phase 6 — Streaming generation tests
# ---------------------------------------------------------------------------


def test_bedrock_stream_no_hits_yields_default() -> None:
    generator = BedrockConverseAnswerGenerator(Settings())
    chunks = list(generator.generate_stream("query", []))
    assert len(chunks) == 1
    assert "could not find grounded passages" in chunks[0].lower()


def test_bedrock_stream_yields_chunks(monkeypatch) -> None:
    generator = BedrockConverseAnswerGenerator(Settings(RAG_AWS_REGION="ap-southeast-2"))

    class FakeClient:
        def converse_stream(self, **kwargs):
            return {
                "stream": [
                    {"contentBlockDelta": {"delta": {"text": "hello "}}},
                    {"contentBlockDelta": {"delta": {"text": "world"}}},
                ]
            }

    monkeypatch.setattr(generator, "_get_client", lambda: FakeClient())
    chunks = list(generator.generate_stream("query", _sample_hits()))
    assert chunks == ["hello ", "world"]


def test_qwen_stream_no_hits_yields_default() -> None:
    class FakeQwenClient:
        def is_configured(self):
            return True

        def stream_chat(self, system_prompt, user_prompt):
            return iter(["should not be called"])

    generator = QwenAnswerGenerator(FakeQwenClient(), Settings())
    chunks = list(generator.generate_stream("query", []))
    assert len(chunks) == 1
    assert "could not find grounded passages" in chunks[0].lower()


def test_qwen_stream_delegates_to_client() -> None:
    class FakeQwenClient:
        def is_configured(self):
            return True

        def stream_chat(self, system_prompt, user_prompt):
            return iter(["qwen-chunk-1", "qwen-chunk-2"])

    generator = QwenAnswerGenerator(FakeQwenClient(), Settings())
    chunks = list(generator.generate_stream("query", _sample_hits()))
    assert chunks == ["qwen-chunk-1", "qwen-chunk-2"]


def test_routed_stream_prefers_qwen() -> None:
    class FakeBedrock:
        def generate_stream(self, **kwargs):
            return iter(["bedrock-chunk"])

    class FakeQwen:
        def is_available(self):
            return True

        def generate_stream(self, **kwargs):
            return iter(["qwen-chunk"])

    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    stream, model = router.generate_stream(
        query="q", hits=_sample_hits(), preferred_model="qwen-plus"
    )
    assert list(stream) == ["qwen-chunk"]
    assert model == "qwen-plus"


def test_routed_stream_fallback_qwen_to_bedrock() -> None:
    class FakeBedrock:
        def generate_stream(self, **kwargs):
            return iter(["bedrock-fallback"])

    class FakeQwen:
        def is_available(self):
            return True

        def generate_stream(self, **kwargs):
            raise RuntimeError("qwen stream failed")

    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    stream, model = router.generate_stream(
        query="q", hits=_sample_hits(), preferred_model="qwen-plus"
    )
    assert list(stream) == ["bedrock-fallback"]
    assert model == "nova-lite"


def test_routed_stream_nova_fallback_to_qwen() -> None:
    class FakeBedrock:
        def generate_stream(self, **kwargs):
            raise RuntimeError("bedrock stream failed")

    class FakeQwen:
        def is_available(self):
            return True

        def generate_stream(self, **kwargs):
            return iter(["qwen-fallback"])

    router = RoutedAnswerGenerator(FakeBedrock(), FakeQwen())
    stream, model = router.generate_stream(
        query="q", hits=_sample_hits(), preferred_model="nova-lite"
    )
    assert list(stream) == ["qwen-fallback"]
    assert model == "qwen-plus"
