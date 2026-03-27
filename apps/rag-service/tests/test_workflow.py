from __future__ import annotations

from app.answer_generator import ModelRoute
from app.config import Settings
from app.models import KeywordResult
from app.workflow import RagWorkflow


class FakeRepository:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.last_request = None

    def retrieve(self, request):
        self.last_request = request
        return self._hits


class FakeQueryProcessor:
    def __init__(
        self,
        intent: str = "factual",
        complexity: str = "medium",
        rewrite: str = "",
        embedding: list[float] | None = None,
    ) -> None:
        self.intent = intent
        self.complexity = complexity
        self.rewrite = rewrite
        self.embedding = embedding

    def detect_intent(self, query: str):
        return {"intent": self.intent, "complexity": self.complexity}

    def extract_keywords(self, query: str):
        return KeywordResult()

    def rewrite_query(self, query: str, intent: str, complexity: str, ll_keywords: list[str] | None = None) -> str:
        return self.rewrite or query

    def build_query_embedding(self, query: str) -> list[float] | None:
        return self.embedding


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], ModelRoute]] = []

    def generate(self, query: str, hits: list[dict], preferred_model: ModelRoute):
        self.calls.append((query, hits, preferred_model))
        return ("final-answer", preferred_model)


class FakeReranker:
    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        return hits[:top_k]


def _sample_hits(score: float = 0.2) -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "chunk_text": "snippet",
            "score": score,
            "citation": {
                "title": "Doc",
                "url": "https://example.com",
                "year": 2025,
                "month": 1,
            },
        }
    ]


def test_workflow_uses_rewritten_query_for_retrieval() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(rewrite="rewritten query")
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    state = workflow.run(query="original query", top_k=5, filters={})
    assert repo.last_request is not None
    assert repo.last_request.query == "rewritten query"
    assert state["answer"] == "final-answer"
    assert state["citations"][0]["sourceId"] == "c1"


def test_workflow_routes_to_qwen_for_high_complexity() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(complexity="high")
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    state = workflow.run(query="hard query", top_k=5, filters={})
    assert state["preferred_model"] == "qwen-plus"
    assert state["answer_model"] == "qwen-plus"


def test_workflow_routes_to_qwen_for_weak_evidence() -> None:
    repo = FakeRepository(_sample_hits(score=0.001))
    qp = FakeQueryProcessor(complexity="medium")
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(RAG_ROUTE_TOP_SCORE_THRESHOLD="0.01"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["preferred_model"] == "qwen-plus"


def test_workflow_routes_to_nova_for_simple_query_with_good_hits() -> None:
    repo = FakeRepository(_sample_hits(score=0.3))
    qp = FakeQueryProcessor(complexity="medium")
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(
            RAG_ROUTE_TOP_SCORE_THRESHOLD="0.01",
            RAG_ROUTE_MIN_HITS="1",
            RAG_ROUTE_COMPLEX_QUERY_TOKEN_THRESHOLD="20",
        ),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    state = workflow.run(query="short query", top_k=5, filters={})
    assert state["preferred_model"] == "nova-lite"
    assert state["answer_model"] == "nova-lite"


def test_workflow_includes_query_embedding_in_request_when_available() -> None:
    repo = FakeRepository(_sample_hits(score=0.3))
    qp = FakeQueryProcessor(embedding=[0.1, 0.2, 0.3])
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    workflow.run(query="embedding query", top_k=5, filters={})
    assert repo.last_request is not None
    assert repo.last_request.query_embedding == [0.1, 0.2, 0.3]


def test_workflow_uses_reranked_hits_for_answer() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor()
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    state = workflow.run(query="query", top_k=1, filters={})
    assert "reranked_hits" in state
    assert len(state["reranked_hits"]) == 1
    assert state["citations"][0]["sourceId"] == "c1"
