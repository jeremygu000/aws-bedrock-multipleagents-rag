"""LangGraph workflow for retrieval, routing, and grounded answer generation."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answer_generator import ModelRoute, RoutedAnswerGenerator
from .config import Settings
from .graph_retriever import GraphRetriever
from .hybrid_fusion import fuse_graph_and_traditional
from .models import GraphContext, RetrievalMode, RetrieveRequest
from .query_processing import QueryProcessor
from .repository import PostgresRepository
from .reranker import LLMReranker

logger = logging.getLogger(__name__)


class RagWorkflowState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    query: str
    top_k: int
    filters: dict[str, Any]
    intent: str
    complexity: str
    hl_keywords: list[str]
    ll_keywords: list[str]
    retrieval_mode: str
    rewritten_query: str
    query_embedding: list[float] | None
    request: RetrieveRequest
    hits: list[dict[str, Any]]
    graph_context: GraphContext
    fused_hits: list[dict[str, Any]]
    reranked_hits: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    preferred_model: ModelRoute
    answer_model: ModelRoute
    answer: str


class RagWorkflow:
    """Compiled workflow for intent/rewrite/retrieve/answer stages."""

    def __init__(
        self,
        settings: Settings,
        repository: PostgresRepository,
        query_processor: QueryProcessor,
        answer_generator: RoutedAnswerGenerator,
        reranker: LLMReranker,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        """Create and compile graph with injected services."""

        self._settings = settings
        self._repository = repository
        self._query_processor = query_processor
        self._answer_generator = answer_generator
        self._reranker = reranker
        self._graph_retriever = graph_retriever

        graph = StateGraph(RagWorkflowState)

        graph.add_node("detect_intent", self._node_detect_intent)
        graph.add_node("extract_keywords", self._node_extract_keywords)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("rewrite_query", self._node_rewrite_query)
        graph.add_node("build_request", self._node_build_request)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("graph_retrieve", self._node_graph_retrieve)
        graph.add_node("fuse", self._node_fuse)
        graph.add_node("rerank", self._node_rerank)
        graph.add_node("build_citations", self._node_build_citations)
        graph.add_node("choose_model", self._node_choose_model)
        graph.add_node("generate_answer", self._node_generate_answer)

        graph.add_edge(START, "detect_intent")
        graph.add_edge("detect_intent", "extract_keywords")
        graph.add_edge("extract_keywords", "determine_mode")
        graph.add_edge("determine_mode", "rewrite_query")
        graph.add_edge("rewrite_query", "build_request")
        graph.add_edge("build_request", "retrieve")
        graph.add_edge("retrieve", "graph_retrieve")
        graph.add_edge("graph_retrieve", "fuse")
        graph.add_edge("fuse", "rerank")
        graph.add_edge("rerank", "build_citations")
        graph.add_edge("build_citations", "choose_model")
        graph.add_edge("choose_model", "generate_answer")
        graph.add_edge("generate_answer", END)

        self._graph = graph.compile()

    def run(self, query: str, top_k: int, filters: dict[str, Any]) -> RagWorkflowState:
        """Execute workflow and return final state."""

        initial_state: RagWorkflowState = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        return self._graph.invoke(initial_state)

    def _node_detect_intent(self, state: RagWorkflowState) -> RagWorkflowState:
        detection = self._query_processor.detect_intent(state["query"])
        return {
            "intent": str(detection.get("intent", "factual")),
            "complexity": str(detection.get("complexity", "medium")),
        }

    def _node_extract_keywords(self, state: RagWorkflowState) -> RagWorkflowState:
        result = self._query_processor.extract_keywords(state["query"])
        return {"hl_keywords": result.hl_keywords, "ll_keywords": result.ll_keywords}

    def _node_determine_mode(self, state: RagWorkflowState) -> RagWorkflowState:
        mode = self._query_processor.determine_retrieval_mode(
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            hl_keywords=state.get("hl_keywords", []),
            ll_keywords=state.get("ll_keywords", []),
        )
        return {"retrieval_mode": mode.value}

    def _node_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
        rewritten = self._query_processor.rewrite_query(
            query=state["query"],
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            ll_keywords=state.get("ll_keywords", []),
        )
        return {"rewritten_query": rewritten}

    def _node_build_request(self, state: RagWorkflowState) -> RagWorkflowState:
        top_k = state["top_k"]
        retrieval_query = state.get("rewritten_query") or state["query"]
        query_embedding = self._query_processor.build_query_embedding(retrieval_query)
        request = RetrieveRequest(
            query=retrieval_query,
            query_embedding=query_embedding,
            k_sparse=max(top_k * 4, 20),
            k_dense=max(top_k * 4, 20),
            k_final=top_k,
            filters=state["filters"],
        )
        return {"request": request, "query_embedding": query_embedding}

    def _node_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        hits = self._repository.retrieve(state["request"])
        return {"hits": hits}

    def _node_graph_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        mode_str = state.get("retrieval_mode", "mix")
        try:
            mode = RetrievalMode(mode_str)
        except ValueError:
            mode = RetrievalMode.MIX

        if mode.skip_graph or not self._graph_retriever:
            return {"graph_context": GraphContext()}

        strategy = mode.graph_strategy or "hybrid"
        query = state.get("rewritten_query") or state["query"]

        try:
            graph_context = self._graph_retriever.retrieve(query, strategy=strategy)
        except Exception:
            logger.exception("Graph retrieval failed, falling back to empty context")
            graph_context = GraphContext()

        return {"graph_context": graph_context}

    def _node_fuse(self, state: RagWorkflowState) -> RagWorkflowState:
        """Fuse traditional retrieval hits with graph-derived chunk hits via weighted RRF."""

        graph_context: GraphContext = state.get("graph_context", GraphContext())
        traditional_hits = state.get("hits", [])

        if graph_context.is_empty or not graph_context.source_chunk_ids:
            return {"fused_hits": traditional_hits}

        try:
            graph_chunk_hits = self._repository.get_chunks_by_ids(graph_context.source_chunk_ids)
        except Exception:
            logger.exception("Failed to fetch graph chunk hits, skipping fusion")
            return {"fused_hits": traditional_hits}

        if not graph_chunk_hits:
            return {"fused_hits": traditional_hits}

        top_k = state.get("top_k", 8)
        fused = fuse_graph_and_traditional(
            traditional_hits=traditional_hits,
            graph_chunk_hits=graph_chunk_hits,
            graph_context=graph_context,
            graph_weight=self._settings.graph_retrieval_weight,
            rrf_k=self._settings.default_rrf_k,
            k_final=top_k,
        )
        return {"fused_hits": fused}

    def _node_rerank(self, state: RagWorkflowState) -> RagWorkflowState:
        hits = state.get("fused_hits") or state.get("hits", [])
        top_k = state.get("top_k", 8)
        reranked = self._reranker.rerank(
            query=state.get("rewritten_query") or state["query"],
            hits=hits,
            top_k=top_k,
        )
        return {"reranked_hits": reranked}

    def _node_build_citations(self, state: RagWorkflowState) -> RagWorkflowState:
        citations = [
            {
                "sourceId": hit["chunk_id"],
                "title": hit["citation"]["title"],
                "url": hit["citation"]["url"],
                "snippet": hit["chunk_text"][:280],
            }
            for hit in state.get("reranked_hits") or state.get("hits", [])
        ]
        return {"citations": citations}

    def _node_choose_model(self, state: RagWorkflowState) -> RagWorkflowState:
        hits = state.get("reranked_hits") or state.get("hits", [])
        top_score = float(hits[0]["score"]) if hits else 0.0
        complexity = state.get("complexity", "medium")
        token_count = len((state.get("rewritten_query") or state["query"]).split())
        weak_evidence = len(hits) < self._settings.route_min_hits or (
            top_score < self._settings.route_top_score_threshold
        )
        complex_query = complexity == "high" or (
            token_count >= self._settings.route_complex_query_token_threshold
        )

        preferred_model: ModelRoute = (
            "qwen-plus" if (complex_query or weak_evidence) else "nova-lite"
        )
        return {"preferred_model": preferred_model}

    def _node_generate_answer(self, state: RagWorkflowState) -> RagWorkflowState:
        hl = state.get("hl_keywords", [])
        ll = state.get("ll_keywords", [])
        keywords = list(dict.fromkeys(hl + ll))
        graph_context: GraphContext | None = state.get("graph_context")
        answer, used_model = self._answer_generator.generate(
            query=state["query"],
            hits=state.get("reranked_hits") or state.get("hits", []),
            preferred_model=state.get("preferred_model", "nova-lite"),
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            keywords=keywords or None,
            graph_context=graph_context,
        )
        return {"answer": answer, "answer_model": used_model}
