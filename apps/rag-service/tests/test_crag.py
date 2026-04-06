from __future__ import annotations

from typing import Any
from unittest.mock import patch

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


def _mock_bedrock_chat_factory(responses: list[str]) -> Any:
    call_index = 0

    def _mock_bedrock_chat(_client: Any, _model_id: str, _sys: str, _user: str, max_tokens: int = 200) -> str:
        nonlocal call_index
        if call_index < len(responses):
            resp = responses[call_index]
            call_index += 1
            return resp
        return '{"relevant": true}'

    return _mock_bedrock_chat


class TestRetrievalGrader:
    @patch("app.crag._bedrock_chat", return_value='{"relevant": true}')
    @patch("app.crag.boto3")
    def test_grade_relevant_document(self, _mock_boto, mock_chat: Any) -> None:
        grader = RetrievalGrader(Settings())
        assert grader.grade("What is APRA?", "APRA is a music rights organization.") is True

    @patch("app.crag._bedrock_chat", return_value='{"relevant": false}')
    @patch("app.crag.boto3")
    def test_grade_irrelevant_document(self, _mock_boto, mock_chat: Any) -> None:
        grader = RetrievalGrader(Settings())
        assert grader.grade("What is APRA?", "The weather today is sunny.") is False

    @patch("app.crag._bedrock_chat", return_value="yes this is relevant true")
    @patch("app.crag.boto3")
    def test_grade_fallback_on_malformed_json(self, _mock_boto, mock_chat: Any) -> None:
        grader = RetrievalGrader(Settings())
        assert grader.grade("query", "doc") is True

    def test_grade_hits_all_relevant_returns_correct(self) -> None:
        mock_chat = _mock_bedrock_chat_factory(['{"relevant": true}', '{"relevant": true}'])
        with patch("app.crag._bedrock_chat", side_effect=mock_chat), patch("app.crag.boto3"):
            grader = RetrievalGrader(
                Settings(
                    RAG_CRAG_UPPER_THRESHOLD="0.7",
                    RAG_CRAG_LOWER_THRESHOLD="0.3",
                    RAG_CRAG_MIN_RELEVANT_DOCS="1",
                ),
            )
            hits = _sample_hits(count=2)
            relevant, verdict = grader.grade_hits("query", hits)
            assert verdict == "correct"
            assert len(relevant) == 2

    def test_grade_hits_none_relevant_returns_incorrect(self) -> None:
        mock_chat = _mock_bedrock_chat_factory(['{"relevant": false}', '{"relevant": false}'])
        with patch("app.crag._bedrock_chat", side_effect=mock_chat), patch("app.crag.boto3"):
            grader = RetrievalGrader(
                Settings(
                    RAG_CRAG_UPPER_THRESHOLD="0.7",
                    RAG_CRAG_LOWER_THRESHOLD="0.3",
                ),
            )
            hits = _sample_hits(count=2)
            relevant, verdict = grader.grade_hits("query", hits)
            assert verdict == "incorrect"
            assert len(relevant) == 0

    def test_grade_hits_partial_returns_ambiguous(self) -> None:
        responses = ['{"relevant": true}', '{"relevant": false}', '{"relevant": false}']
        mock_chat = _mock_bedrock_chat_factory(responses)
        with patch("app.crag._bedrock_chat", side_effect=mock_chat), patch("app.crag.boto3"):
            grader = RetrievalGrader(
                Settings(
                    RAG_CRAG_UPPER_THRESHOLD="0.7",
                    RAG_CRAG_LOWER_THRESHOLD="0.3",
                    RAG_CRAG_MIN_RELEVANT_DOCS="1",
                ),
            )
            hits = _sample_hits(count=3)
            relevant, verdict = grader.grade_hits("query", hits)
            assert verdict == "ambiguous"
            assert len(relevant) == 1

    @patch("app.crag.boto3")
    def test_grade_hits_empty_returns_incorrect(self, _mock_boto: Any) -> None:
        grader = RetrievalGrader(Settings())
        relevant, verdict = grader.grade_hits("query", [])
        assert verdict == "incorrect"
        assert relevant == []


class TestCragQueryRewriter:
    @patch("app.crag._bedrock_chat", return_value="optimized web search query")
    @patch("app.crag.boto3")
    def test_rewrite_returns_llm_output(self, _mock_boto, mock_chat: Any) -> None:
        rewriter = CragQueryRewriter(Settings())
        result = rewriter.rewrite("original query")
        assert result == "optimized web search query"

    @patch("app.crag._bedrock_chat", return_value="")
    @patch("app.crag.boto3")
    def test_rewrite_fallback_on_empty_response(self, _mock_boto, mock_chat: Any) -> None:
        rewriter = CragQueryRewriter(Settings())
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
        mock_chat = _mock_bedrock_chat_factory(['{"relevant": true}'])
        with patch("app.crag._bedrock_chat", side_effect=mock_chat), patch("app.crag.boto3"):
            grader = RetrievalGrader(
                Settings(
                    RAG_CRAG_UPPER_THRESHOLD="0.7",
                    RAG_CRAG_LOWER_THRESHOLD="0.3",
                    RAG_CRAG_MIN_RELEVANT_DOCS="1",
                ),
            )
            workflow = _build_workflow(enable_crag=True, grader=grader)
            state = workflow.run(query="query", top_k=5, filters={})
            assert state["retrieval_verdict"] == "correct"
            assert state["answer"] == "final-answer"
            assert state.get("crag_rewritten_query") is None

    def test_incorrect_verdict_triggers_web_search_path(self) -> None:
        grader_mock = _mock_bedrock_chat_factory(['{"relevant": false}'])
        rewriter_mock = _mock_bedrock_chat_factory(["web optimized query"])
        with (
            patch("app.crag._bedrock_chat", side_effect=grader_mock),
            patch("app.crag.boto3"),
        ):
            grader = RetrievalGrader(
                Settings(
                    RAG_CRAG_UPPER_THRESHOLD="0.7",
                    RAG_CRAG_LOWER_THRESHOLD="0.3",
                ),
            )
        with (
            patch("app.crag._bedrock_chat", side_effect=rewriter_mock),
            patch("app.crag.boto3"),
        ):
            rewriter = CragQueryRewriter(Settings())
        web_searcher = CragWebSearcher(Settings(RAG_CRAG_ENABLE_WEB_SEARCH="false"))

        call_index = [0]
        responses = ['{"relevant": false}', "web optimized query"]

        def combined_mock(_c: Any, _m: str, _s: str, _u: str, max_tokens: int = 200) -> str:
            idx = call_index[0]
            call_index[0] += 1
            return responses[idx] if idx < len(responses) else ""

        with patch("app.crag._bedrock_chat", side_effect=combined_mock), patch("app.crag.boto3"):
            grader = RetrievalGrader(
                Settings(RAG_CRAG_UPPER_THRESHOLD="0.7", RAG_CRAG_LOWER_THRESHOLD="0.3"),
            )
            rewriter = CragQueryRewriter(Settings())
            workflow = _build_workflow(
                enable_crag=True, grader=grader, rewriter=rewriter, web_searcher=web_searcher,
            )
            state = workflow.run(query="query", top_k=5, filters={})
            assert state["retrieval_verdict"] == "incorrect"
            assert state["crag_rewritten_query"] == "web optimized query"
            assert state["answer"] == "final-answer"

    def test_ambiguous_verdict_triggers_web_search_path(self) -> None:
        call_index = [0]
        responses = ['{"relevant": true}', '{"relevant": false}', '{"relevant": false}', "better query"]

        def combined_mock(_c: Any, _m: str, _s: str, _u: str, max_tokens: int = 200) -> str:
            idx = call_index[0]
            call_index[0] += 1
            return responses[idx] if idx < len(responses) else ""

        with patch("app.crag._bedrock_chat", side_effect=combined_mock), patch("app.crag.boto3"):
            grader = RetrievalGrader(
                Settings(
                    RAG_CRAG_UPPER_THRESHOLD="0.7",
                    RAG_CRAG_LOWER_THRESHOLD="0.3",
                    RAG_CRAG_MIN_RELEVANT_DOCS="1",
                ),
            )
            rewriter = CragQueryRewriter(Settings())
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
