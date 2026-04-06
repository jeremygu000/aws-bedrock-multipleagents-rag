from __future__ import annotations

from typing import Any

from app.config import Settings
from app.crag import CragQueryRewriter, CragWebSearcher, RetrievalGrader
from app.models import KeywordResult, RetrievalMode
from app.workflow import RagWorkflow


class FakeRepository:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.last_request = None

    def retrieve(self, request: Any) -> list[dict]:
        self.last_request = request
        return self._hits

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        return []


class FakeQueryProcessor:
    def __init__(self, rewrite: str = "") -> None:
        self.rewrite = rewrite

    def detect_intent(self, query: str) -> dict:
        return {"intent": "factual", "complexity": "medium"}

    def extract_keywords(self, query: str) -> KeywordResult:
        return KeywordResult()

    def rewrite_query(
        self, query: str, intent: str, complexity: str, ll_keywords: list[str] | None = None
    ) -> str:
        return self.rewrite or query

    def build_query_embedding(self, query: str) -> list[float] | None:
        return None

    def determine_retrieval_mode(
        self, intent: str, complexity: str, hl_keywords: list[str], ll_keywords: list[str]
    ) -> RetrievalMode:
        return RetrievalMode.NAIVE


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def generate(self, query: str, hits: list[dict], preferred_model: str, **kwargs: Any) -> tuple:
        self.calls.append((query, hits, preferred_model))
        return ("final-answer", preferred_model)


class FakeReranker:
    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        return hits[:top_k]


class FakeQwenClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_index = 0
        self.chat_calls: list[tuple[str, str]] = []

    def is_configured(self) -> bool:
        return True

    def chat(self, system_prompt: str, user_prompt: str, *, max_tokens: int | None = None) -> str:
        self.chat_calls.append((system_prompt, user_prompt))
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return '{"relevant": true}'


def _sample_hits(score: float = 0.2, count: int = 1) -> list[dict]:
    hits = []
    for i in range(count):
        hits.append(
            {
                "chunk_id": f"c{i + 1}",
                "chunk_text": f"snippet {i + 1}",
                "score": score,
                "citation": {
                    "title": "Doc",
                    "url": "https://example.com",
                    "year": 2025,
                    "month": 1,
                },
            }
        )
    return hits


def _build_workflow(
    hits: list[dict] | None = None,
    enable_crag: bool = False,
    grader: RetrievalGrader | None = None,
    rewriter: CragQueryRewriter | None = None,
    web_searcher: CragWebSearcher | None = None,
) -> RagWorkflow:
    return RagWorkflow(
        settings=Settings(RAG_ENABLE_CRAG=str(enable_crag).lower()),
        repository=FakeRepository(hits or _sample_hits()),
        query_processor=FakeQueryProcessor(),
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        retrieval_grader=grader,
        crag_query_rewriter=rewriter,
        crag_web_searcher=web_searcher,
    )


class TestRetrievalGrader:
    def test_grade_relevant_document(self) -> None:
        qwen = FakeQwenClient(responses=['{"relevant": true}'])
        grader = RetrievalGrader(Settings(), qwen)
        assert grader.grade("What is APRA?", "APRA is a music rights organization.") is True

    def test_grade_irrelevant_document(self) -> None:
        qwen = FakeQwenClient(responses=['{"relevant": false}'])
        grader = RetrievalGrader(Settings(), qwen)
        assert grader.grade("What is APRA?", "The weather today is sunny.") is False

    def test_grade_fallback_on_malformed_json(self) -> None:
        qwen = FakeQwenClient(responses=["yes this is relevant true"])
        grader = RetrievalGrader(Settings(), qwen)
        assert grader.grade("query", "doc") is True

    def test_grade_hits_all_relevant_returns_correct(self) -> None:
        qwen = FakeQwenClient(responses=['{"relevant": true}', '{"relevant": true}'])
        grader = RetrievalGrader(
            Settings(
                RAG_CRAG_UPPER_THRESHOLD="0.7",
                RAG_CRAG_LOWER_THRESHOLD="0.3",
                RAG_CRAG_MIN_RELEVANT_DOCS="1",
            ),
            qwen,
        )
        hits = _sample_hits(count=2)
        relevant, verdict = grader.grade_hits("query", hits)
        assert verdict == "correct"
        assert len(relevant) == 2

    def test_grade_hits_none_relevant_returns_incorrect(self) -> None:
        qwen = FakeQwenClient(responses=['{"relevant": false}', '{"relevant": false}'])
        grader = RetrievalGrader(
            Settings(
                RAG_CRAG_UPPER_THRESHOLD="0.7",
                RAG_CRAG_LOWER_THRESHOLD="0.3",
            ),
            qwen,
        )
        hits = _sample_hits(count=2)
        relevant, verdict = grader.grade_hits("query", hits)
        assert verdict == "incorrect"
        assert len(relevant) == 0

    def test_grade_hits_partial_returns_ambiguous(self) -> None:
        qwen = FakeQwenClient(
            responses=['{"relevant": true}', '{"relevant": false}', '{"relevant": false}']
        )
        grader = RetrievalGrader(
            Settings(
                RAG_CRAG_UPPER_THRESHOLD="0.7",
                RAG_CRAG_LOWER_THRESHOLD="0.3",
                RAG_CRAG_MIN_RELEVANT_DOCS="1",
            ),
            qwen,
        )
        hits = _sample_hits(count=3)
        relevant, verdict = grader.grade_hits("query", hits)
        assert verdict == "ambiguous"
        assert len(relevant) == 1

    def test_grade_hits_empty_returns_incorrect(self) -> None:
        qwen = FakeQwenClient()
        grader = RetrievalGrader(Settings(), qwen)
        relevant, verdict = grader.grade_hits("query", [])
        assert verdict == "incorrect"
        assert relevant == []


class TestCragQueryRewriter:
    def test_rewrite_returns_llm_output(self) -> None:
        qwen = FakeQwenClient(responses=["optimized web search query"])
        rewriter = CragQueryRewriter(Settings(), qwen)
        result = rewriter.rewrite("original query")
        assert result == "optimized web search query"

    def test_rewrite_fallback_on_empty_response(self) -> None:
        qwen = FakeQwenClient(responses=[""])
        rewriter = CragQueryRewriter(Settings(), qwen)
        result = rewriter.rewrite("original query")
        assert result == "original query"


class TestCragWebSearcher:
    def test_search_disabled_returns_empty(self) -> None:
        searcher = CragWebSearcher(Settings(RAG_CRAG_ENABLE_WEB_SEARCH="false"))
        assert searcher.search("query") == []

    def test_search_no_api_key_returns_empty(self) -> None:
        searcher = CragWebSearcher(
            Settings(RAG_CRAG_ENABLE_WEB_SEARCH="true", TAVILY_API_KEY="")
        )
        assert searcher.search("query") == []


class TestWorkflowCragDisabled:
    def test_crag_disabled_passes_through(self) -> None:
        workflow = _build_workflow(enable_crag=False)
        state = workflow.run(query="query", top_k=5, filters={})
        assert state["answer"] == "final-answer"
        assert state.get("retrieval_verdict") == "correct"

    def test_crag_disabled_no_grader_passes_through(self) -> None:
        workflow = _build_workflow(enable_crag=True, grader=None)
        state = workflow.run(query="query", top_k=5, filters={})
        assert state["answer"] == "final-answer"
        assert state.get("retrieval_verdict") == "correct"


class TestWorkflowCragEnabled:
    def test_correct_verdict_skips_web_search(self) -> None:
        qwen = FakeQwenClient(responses=['{"relevant": true}'])
        grader = RetrievalGrader(
            Settings(
                RAG_CRAG_UPPER_THRESHOLD="0.7",
                RAG_CRAG_LOWER_THRESHOLD="0.3",
                RAG_CRAG_MIN_RELEVANT_DOCS="1",
            ),
            qwen,
        )
        workflow = _build_workflow(enable_crag=True, grader=grader)
        state = workflow.run(query="query", top_k=5, filters={})
        assert state["retrieval_verdict"] == "correct"
        assert state["answer"] == "final-answer"
        assert state.get("crag_rewritten_query") is None

    def test_incorrect_verdict_triggers_web_search_path(self) -> None:
        qwen = FakeQwenClient(responses=['{"relevant": false}', "rewritten query for web"])
        grader = RetrievalGrader(
            Settings(
                RAG_CRAG_UPPER_THRESHOLD="0.7",
                RAG_CRAG_LOWER_THRESHOLD="0.3",
            ),
            qwen,
        )
        rewriter_qwen = FakeQwenClient(responses=["web optimized query"])
        rewriter = CragQueryRewriter(Settings(), rewriter_qwen)
        web_searcher = CragWebSearcher(Settings(RAG_CRAG_ENABLE_WEB_SEARCH="false"))

        workflow = _build_workflow(
            enable_crag=True,
            grader=grader,
            rewriter=rewriter,
            web_searcher=web_searcher,
        )
        state = workflow.run(query="query", top_k=5, filters={})
        assert state["retrieval_verdict"] == "incorrect"
        assert state["crag_rewritten_query"] == "web optimized query"
        assert state["answer"] == "final-answer"

    def test_ambiguous_verdict_triggers_web_search_path(self) -> None:
        responses = ['{"relevant": true}', '{"relevant": false}', '{"relevant": false}']
        qwen = FakeQwenClient(responses=responses)
        grader = RetrievalGrader(
            Settings(
                RAG_CRAG_UPPER_THRESHOLD="0.7",
                RAG_CRAG_LOWER_THRESHOLD="0.3",
                RAG_CRAG_MIN_RELEVANT_DOCS="1",
            ),
            qwen,
        )
        rewriter_qwen = FakeQwenClient(responses=["better query"])
        rewriter = CragQueryRewriter(Settings(), rewriter_qwen)
        web_searcher = CragWebSearcher(Settings(RAG_CRAG_ENABLE_WEB_SEARCH="false"))

        workflow = _build_workflow(
            hits=_sample_hits(count=3),
            enable_crag=True,
            grader=grader,
            rewriter=rewriter,
            web_searcher=web_searcher,
        )
        state = workflow.run(query="query", top_k=5, filters={})
        assert state["retrieval_verdict"] == "ambiguous"
        assert state["crag_rewritten_query"] == "better query"
        assert state["answer"] == "final-answer"

    def test_existing_tests_unaffected_when_crag_disabled(self) -> None:
        workflow = _build_workflow(
            hits=_sample_hits(score=0.3),
            enable_crag=False,
        )
        state = workflow.run(query="simple query", top_k=5, filters={})
        assert state["answer"] == "final-answer"
        assert state["citations"][0]["sourceId"] == "c1"
        assert state.get("retrieval_verdict") == "correct"
