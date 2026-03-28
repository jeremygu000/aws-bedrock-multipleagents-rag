from __future__ import annotations

from app.answer_generator import ModelRoute
from app.config import Settings
from app.models import GraphContext, GraphEntity, GraphRelation, KeywordResult, RetrievalMode
from app.workflow import RagWorkflow


class FakeRepository:
    def __init__(self, hits: list[dict], chunk_map: dict[str, dict] | None = None) -> None:
        self._hits = hits
        self._chunk_map = chunk_map or {}
        self.last_request = None
        self.get_chunks_by_ids_calls: list[list[str]] = []

    def retrieve(self, request):
        self.last_request = request
        return self._hits

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        self.get_chunks_by_ids_calls.append(chunk_ids)
        return [self._chunk_map[cid] for cid in chunk_ids if cid in self._chunk_map]


class FakeQueryProcessor:
    def __init__(
        self,
        intent: str = "factual",
        complexity: str = "medium",
        rewrite: str = "",
        embedding: list[float] | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.NAIVE,
    ) -> None:
        self.intent = intent
        self.complexity = complexity
        self.rewrite = rewrite
        self.embedding = embedding
        self.retrieval_mode = retrieval_mode

    def detect_intent(self, query: str):
        return {"intent": self.intent, "complexity": self.complexity}

    def extract_keywords(self, query: str):
        return KeywordResult()

    def rewrite_query(
        self, query: str, intent: str, complexity: str, ll_keywords: list[str] | None = None
    ) -> str:
        return self.rewrite or query

    def build_query_embedding(self, query: str) -> list[float] | None:
        return self.embedding

    def determine_retrieval_mode(
        self,
        intent: str,
        complexity: str,
        hl_keywords: list[str],
        ll_keywords: list[str],
    ) -> RetrievalMode:
        return self.retrieval_mode


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], ModelRoute]] = []
        self.last_graph_context: GraphContext | None = None

    def generate(
        self,
        query: str,
        hits: list[dict],
        preferred_model: ModelRoute,
        intent: str = "factual",
        complexity: str = "medium",
        keywords: list[str] | None = None,
        graph_context: GraphContext | None = None,
    ):
        self.calls.append((query, hits, preferred_model))
        self.last_graph_context = graph_context
        return ("final-answer", preferred_model)


class FakeReranker:
    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        return hits[:top_k]


class FakeGraphRetriever:
    def __init__(
        self,
        context: GraphContext | None = None,
        raise_on_call: bool = False,
    ) -> None:
        self._context = context or GraphContext()
        self._raise_on_call = raise_on_call
        self.calls: list[tuple[str, str]] = []

    def retrieve(self, query: str, strategy: str = "hybrid") -> GraphContext:
        self.calls.append((query, strategy))
        if self._raise_on_call:
            raise RuntimeError("graph retriever boom")
        return self._context


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


# ---------------------------------------------------------------------------
# determine_mode + graph_retrieve workflow integration tests
# ---------------------------------------------------------------------------


def test_workflow_sets_retrieval_mode_in_state() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.LOCAL)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["retrieval_mode"] == "local"


def test_workflow_graph_retrieve_skipped_when_no_retriever() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.MIX)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=None,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["graph_context"].is_empty
    assert state["answer"] == "final-answer"


def test_workflow_graph_retrieve_skipped_when_naive_mode() -> None:
    gr = FakeGraphRetriever()
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.NAIVE)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["graph_context"].is_empty
    assert len(gr.calls) == 0


def test_workflow_graph_retrieve_calls_retriever_with_correct_strategy() -> None:
    graph_ctx = GraphContext(
        entities=[GraphEntity(entity_id="e1", name="Test", type="Work", description="desc")],
        relations=[],
    )
    gr = FakeGraphRetriever(context=graph_ctx)
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.LOCAL)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    state = workflow.run(query="test query", top_k=5, filters={})
    assert len(gr.calls) == 1
    assert gr.calls[0][1] == "local"
    assert not state["graph_context"].is_empty
    assert state["graph_context"].entities[0].name == "Test"


def test_workflow_graph_retrieve_uses_hybrid_for_mix_mode() -> None:
    gr = FakeGraphRetriever()
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.MIX)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    workflow.run(query="query", top_k=5, filters={})
    assert len(gr.calls) == 1
    assert gr.calls[0][1] == "hybrid"


def test_workflow_graph_retrieve_uses_rewritten_query() -> None:
    gr = FakeGraphRetriever()
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(rewrite="rewritten for graph", retrieval_mode=RetrievalMode.GLOBAL)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    workflow.run(query="original query", top_k=5, filters={})
    assert gr.calls[0][0] == "rewritten for graph"
    assert gr.calls[0][1] == "global"


def test_workflow_graph_retrieve_graceful_on_exception() -> None:
    gr = FakeGraphRetriever(raise_on_call=True)
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.HYBRID)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["graph_context"].is_empty
    assert state["answer"] == "final-answer"


def test_workflow_graph_context_preserved_through_pipeline() -> None:
    graph_ctx = GraphContext(
        entities=[
            GraphEntity(entity_id="e1", name="Entity1", type="Person", description="A person")
        ],
        relations=[
            GraphRelation(
                source_entity="Entity1",
                target_entity="Entity2",
                relation_type="WROTE",
                evidence="Entity1 wrote something",
            )
        ],
    )
    gr = FakeGraphRetriever(context=graph_ctx)
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.HYBRID)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert len(state["graph_context"].entities) == 1
    assert len(state["graph_context"].relations) == 1
    assert state["graph_context"].entities[0].name == "Entity1"
    assert state["graph_context"].relations[0].relation_type == "WROTE"


# ---------------------------------------------------------------------------
# fuse node workflow integration tests
# ---------------------------------------------------------------------------


def _graph_chunk(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc1",
        "chunk_text": f"graph text {chunk_id}",
        "score": 0.0,
        "category": "test",
        "lang": "en",
        "source_type": "crawler",
        "metadata": {},
        "citation": {"title": "GraphDoc", "url": "https://graph.com", "year": 2025, "month": 3},
    }


def test_workflow_fuse_passthrough_when_graph_context_empty() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.MIX)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=FakeGraphRetriever(),
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert len(repo.get_chunks_by_ids_calls) == 0
    assert state["fused_hits"][0]["chunk_id"] == "c1"


def test_workflow_fuse_fetches_graph_chunks_and_merges() -> None:
    graph_ctx = GraphContext(
        entities=[GraphEntity(entity_id="e1", name="E1", type="Work", description="desc")],
        relations=[],
        source_chunk_ids=["gc1"],
    )
    chunk_map = {"gc1": _graph_chunk("gc1")}
    repo = FakeRepository(_sample_hits(), chunk_map=chunk_map)
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.LOCAL)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=FakeGraphRetriever(context=graph_ctx),
    )

    state = workflow.run(query="query", top_k=10, filters={})
    assert len(repo.get_chunks_by_ids_calls) == 1
    assert "gc1" in repo.get_chunks_by_ids_calls[0]
    chunk_ids = {h["chunk_id"] for h in state["fused_hits"]}
    assert "c1" in chunk_ids
    assert "gc1" in chunk_ids


def test_workflow_fuse_graceful_when_get_chunks_fails() -> None:
    graph_ctx = GraphContext(
        entities=[],
        relations=[],
        source_chunk_ids=["gc1"],
    )

    class FailingRepo(FakeRepository):
        def get_chunks_by_ids(self, chunk_ids):
            raise RuntimeError("db connection failed")

    repo = FailingRepo(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.HYBRID)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=FakeGraphRetriever(context=graph_ctx),
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["fused_hits"][0]["chunk_id"] == "c1"
    assert state["answer"] == "final-answer"


def test_workflow_fuse_skipped_when_no_source_chunk_ids() -> None:
    graph_ctx = GraphContext(
        entities=[GraphEntity(entity_id="e1", name="E1", type="Work", description="desc")],
        relations=[],
        source_chunk_ids=[],
    )
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.GLOBAL)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=FakeGraphRetriever(context=graph_ctx),
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert len(repo.get_chunks_by_ids_calls) == 0
    assert state["fused_hits"][0]["chunk_id"] == "c1"


def test_workflow_rerank_uses_fused_hits() -> None:
    graph_ctx = GraphContext(
        entities=[],
        relations=[],
        source_chunk_ids=["gc1"],
    )
    chunk_map = {"gc1": _graph_chunk("gc1")}
    repo = FakeRepository(_sample_hits(), chunk_map=chunk_map)
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.LOCAL)
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=FakeGraphRetriever(context=graph_ctx),
    )

    state = workflow.run(query="query", top_k=10, filters={})
    reranked_ids = {h["chunk_id"] for h in state["reranked_hits"]}
    assert "gc1" in reranked_ids or "c1" in reranked_ids


def test_workflow_fuse_with_naive_mode_skips_graph_entirely() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.NAIVE)
    gr = FakeGraphRetriever()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=FakeAnswerGenerator(),
        reranker=FakeReranker(),
        graph_retriever=gr,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert len(gr.calls) == 0
    assert state["graph_context"].is_empty
    assert len(repo.get_chunks_by_ids_calls) == 0
    assert state["fused_hits"][0]["chunk_id"] == "c1"


# ---------------------------------------------------------------------------
# Phase 3.4 — generate_answer receives graph_context
# ---------------------------------------------------------------------------


def test_workflow_generate_answer_receives_graph_context() -> None:
    graph_ctx = GraphContext(
        entities=[
            GraphEntity(
                entity_id="e1",
                name="TestOrg",
                type="Organization",
                description="A test org",
                score=0.9,
            ),
        ],
        relations=[
            GraphRelation(
                source_entity="TestOrg",
                target_entity="ProjectX",
                relation_type="develops",
                evidence="Develops project X",
                score=0.8,
            ),
        ],
        source_chunk_ids=["gc1"],
    )
    chunk_map = {"gc1": _graph_chunk("gc1")}
    repo = FakeRepository(_sample_hits(), chunk_map=chunk_map)
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.LOCAL)
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        graph_retriever=FakeGraphRetriever(context=graph_ctx),
    )

    state = workflow.run(query="query", top_k=10, filters={})
    assert state["answer"] == "final-answer"
    assert answer_gen.last_graph_context is not None
    assert not answer_gen.last_graph_context.is_empty
    assert answer_gen.last_graph_context.entities[0].name == "TestOrg"


def test_workflow_generate_answer_no_graph_context_when_disabled() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(retrieval_mode=RetrievalMode.NAIVE)
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["answer"] == "final-answer"
    ctx = answer_gen.last_graph_context
    assert ctx is None or ctx.is_empty


# ---------------------------------------------------------------------------
# Phase L2 — query_cache integration tests
# ---------------------------------------------------------------------------


class FakeQueryCache:
    def __init__(
        self,
        hit_result: dict | None = None,
        raise_on_lookup: bool = False,
        raise_on_store: bool = False,
    ) -> None:
        self._hit_result = hit_result
        self._raise_on_lookup = raise_on_lookup
        self._raise_on_store = raise_on_store
        self.lookup_calls: list[list[float]] = []
        self.store_calls: list[dict] = []

    def lookup(self, query_embedding: list[float]) -> dict | None:
        self.lookup_calls.append(query_embedding)
        if self._raise_on_lookup:
            raise RuntimeError("cache db down")
        return self._hit_result

    def store(
        self,
        query_original: str,
        query_rewritten: str | None,
        query_embedding: list[float],
        answer: str,
        citations: list[dict],
        model_used: str,
        source_doc_ids: list[str] | None = None,
    ) -> str:
        if self._raise_on_store:
            raise RuntimeError("store failed")
        self.store_calls.append(
            {
                "query_original": query_original,
                "query_rewritten": query_rewritten,
                "answer": answer,
                "citations": citations,
                "model_used": model_used,
                "source_doc_ids": source_doc_ids,
            }
        )
        return "fake-key"


def test_workflow_cache_miss_runs_full_pipeline() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(embedding=[0.1, 0.2])
    answer_gen = FakeAnswerGenerator()
    cache = FakeQueryCache(hit_result=None)
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="true"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    state = workflow.run(query="test query", top_k=5, filters={})
    assert state["answer"] == "final-answer"
    assert state.get("cache_hit") is False
    assert len(cache.lookup_calls) == 1
    assert len(cache.store_calls) == 1
    assert cache.store_calls[0]["query_original"] == "test query"


def test_workflow_cache_hit_skips_pipeline() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(embedding=[0.1, 0.2])
    answer_gen = FakeAnswerGenerator()
    cached = {
        "answer": "cached answer",
        "citations": [{"sourceId": "cached-c1"}],
        "model_used": "nova-lite",
        "cache_key": "abc",
        "similarity": 0.98,
    }
    cache = FakeQueryCache(hit_result=cached)
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="true"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    state = workflow.run(query="test query", top_k=5, filters={})
    assert state["cache_hit"] is True
    assert state["answer"] == "cached answer"
    assert state["citations"] == [{"sourceId": "cached-c1"}]
    assert state["answer_model"] == "nova-lite"
    assert len(answer_gen.calls) == 0
    assert repo.last_request is None
    assert len(cache.store_calls) == 0


def test_workflow_cache_disabled_skips_lookup() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor()
    answer_gen = FakeAnswerGenerator()
    cache = FakeQueryCache()
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="false"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["answer"] == "final-answer"
    assert len(cache.lookup_calls) == 0
    assert len(cache.store_calls) == 0


def test_workflow_cache_no_cache_object_runs_normally() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor()
    answer_gen = FakeAnswerGenerator()
    workflow = RagWorkflow(
        settings=Settings(),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=None,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["answer"] == "final-answer"
    assert state.get("cache_hit") is False


def test_workflow_cache_lookup_error_continues_pipeline() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(embedding=[0.1])
    answer_gen = FakeAnswerGenerator()
    cache = FakeQueryCache(raise_on_lookup=True)
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="true"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["answer"] == "final-answer"
    assert state.get("cache_hit") is False


def test_workflow_cache_store_error_does_not_fail_pipeline() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(embedding=[0.1])
    answer_gen = FakeAnswerGenerator()
    cache = FakeQueryCache(raise_on_store=True)
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="true"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["answer"] == "final-answer"


def test_workflow_cache_no_embedding_skips_lookup() -> None:
    repo = FakeRepository(_sample_hits())
    qp = FakeQueryProcessor(embedding=None)
    answer_gen = FakeAnswerGenerator()
    cache = FakeQueryCache()
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="true"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    state = workflow.run(query="query", top_k=5, filters={})
    assert state["answer"] == "final-answer"
    assert len(cache.lookup_calls) == 0


def test_workflow_cache_store_captures_source_doc_ids() -> None:
    hits = [
        {
            "chunk_id": "c1",
            "doc_id": "doc-a",
            "chunk_text": "snippet",
            "score": 0.2,
            "citation": {"title": "Doc", "url": "https://example.com", "year": 2025, "month": 1},
        },
        {
            "chunk_id": "c2",
            "doc_id": "doc-b",
            "chunk_text": "snippet2",
            "score": 0.1,
            "citation": {"title": "Doc2", "url": "https://example.com", "year": 2025, "month": 1},
        },
    ]
    repo = FakeRepository(hits)
    qp = FakeQueryProcessor(embedding=[0.1])
    answer_gen = FakeAnswerGenerator()
    cache = FakeQueryCache()
    workflow = RagWorkflow(
        settings=Settings(RAG_ENABLE_QUERY_CACHE="true"),
        repository=repo,
        query_processor=qp,
        answer_generator=answer_gen,
        reranker=FakeReranker(),
        query_cache=cache,
    )

    workflow.run(query="query", top_k=5, filters={})
    assert len(cache.store_calls) == 1
    stored_doc_ids = set(cache.store_calls[0]["source_doc_ids"])
    assert "doc-a" in stored_doc_ids
    assert "doc-b" in stored_doc_ids
