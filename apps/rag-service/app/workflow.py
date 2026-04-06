from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .answer_generator import ModelRoute, RoutedAnswerGenerator
from .config import Settings
from .crag import CragQueryRewriter, CragWebSearcher, RetrievalGrader
from .decomposition_retriever import DecompositionRetriever
from .graph_retriever import GraphRetriever
from .hybrid_fusion import fuse_graph_and_traditional
from .models import GraphContext, RetrievalMode, RetrieveRequest
from .query_cache import QueryCache
from .query_decomposer import DecompositionResult, QueryDecomposer, SubQuestion
from .query_processing import QueryProcessor
from .repository import PostgresRepository
from .reranker import LLMReranker
from .tracing import (
    CACHE_OPERATIONS,
    PIPELINE_NODE_ERRORS,
    PIPELINE_NODE_LATENCY,
    RETRIEVAL_HIT_COUNT,
    get_tracing,
)

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
    cache_hit: bool
    # --- CRAG (Corrective RAG) fields ---
    retrieval_verdict: str  # "correct" | "incorrect" | "ambiguous"
    crag_retry_count: int
    web_search_results: list[dict[str, Any]]
    crag_rewritten_query: str
    # --- HyDE (Hypothetical Document Embeddings) fields ---
    hyde_hypothesis: str | None
    hyde_embeddings: list[list[float]] | None
    hyde_strategy: str  # "enabled" | "skipped" | "disabled" | "fallback"
    # --- Query Decomposition fields ---
    should_decompose: bool
    decomposition_decision: DecompositionResult | None
    sub_questions: list[SubQuestion]
    sub_question_hits_list: list[list[dict[str, Any]]]
    decomposition_used: bool
    # --- Community Detection fields ---
    community_summaries: list[dict[str, Any]]


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
        query_cache: QueryCache | None = None,
        retrieval_grader: RetrievalGrader | None = None,
        crag_query_rewriter: CragQueryRewriter | None = None,
        crag_web_searcher: CragWebSearcher | None = None,
        hyde_retriever=None,
        query_decomposer: QueryDecomposer | None = None,
        decomposition_retriever: DecompositionRetriever | None = None,
        community_store=None,
    ) -> None:
        """Create and compile graph with injected services."""

        self._settings = settings
        self._repository = repository
        self._query_processor = query_processor
        self._answer_generator = answer_generator
        self._reranker = reranker
        self._graph_retriever = graph_retriever
        self._query_cache = query_cache
        self._retrieval_grader = retrieval_grader
        self._crag_query_rewriter = crag_query_rewriter
        self._crag_web_searcher = crag_web_searcher
        self._hyde_retriever = hyde_retriever
        self._query_decomposer = query_decomposer
        self._decomposition_retriever = decomposition_retriever
        self._community_store = community_store

        graph = StateGraph(RagWorkflowState)

        graph.add_node("check_cache", self._node_check_cache)
        graph.add_node("detect_intent", self._node_detect_intent)
        graph.add_node("extract_keywords", self._node_extract_keywords)
        graph.add_node("determine_mode", self._node_determine_mode)
        graph.add_node("rewrite_query", self._node_rewrite_query)
        graph.add_node("build_request", self._node_build_request)
        graph.add_node("decompose_query", self._node_decompose_query)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("graph_retrieve", self._node_graph_retrieve)
        graph.add_node("fuse", self._node_fuse)
        graph.add_node("rerank", self._node_rerank)
        graph.add_node("grade_retrieval", self._node_grade_retrieval)
        graph.add_node("crag_rewrite_query", self._node_crag_rewrite_query)
        graph.add_node("crag_web_search", self._node_crag_web_search)
        graph.add_node("build_citations", self._node_build_citations)
        graph.add_node("choose_model", self._node_choose_model)
        graph.add_node("generate_answer", self._node_generate_answer)
        graph.add_node("store_cache", self._node_store_cache)

        graph.add_edge(START, "check_cache")
        graph.add_conditional_edges(
            "check_cache",
            self._route_after_cache_check,
            {"cache_hit": END, "cache_miss": "detect_intent"},
        )
        graph.add_edge("detect_intent", "extract_keywords")
        graph.add_edge("extract_keywords", "determine_mode")
        graph.add_edge("determine_mode", "rewrite_query")
        graph.add_edge("rewrite_query", "build_request")
        graph.add_conditional_edges(
            "build_request",
            self._route_after_build_request,
            {"decompose": "decompose_query", "retrieve": "retrieve"},
        )
        graph.add_edge("decompose_query", "retrieve")
        graph.add_edge("retrieve", "graph_retrieve")
        graph.add_edge("graph_retrieve", "fuse")
        graph.add_edge("fuse", "rerank")
        graph.add_edge("rerank", "grade_retrieval")
        graph.add_conditional_edges(
            "grade_retrieval",
            self._route_after_grade_retrieval,
            {"correct": "build_citations", "needs_web_search": "crag_rewrite_query"},
        )
        graph.add_edge("crag_rewrite_query", "crag_web_search")
        graph.add_edge("crag_web_search", "build_citations")
        graph.add_edge("build_citations", "choose_model")
        graph.add_edge("choose_model", "generate_answer")
        graph.add_edge("generate_answer", "store_cache")
        graph.add_edge("store_cache", END)

        self._graph = graph.compile()

        # Build a second graph that stops after choose_model (for streaming).
        pre_gen_graph = StateGraph(RagWorkflowState)
        for name in [
            "check_cache",
            "detect_intent",
            "extract_keywords",
            "determine_mode",
            "rewrite_query",
            "build_request",
            "decompose_query",
            "retrieve",
            "graph_retrieve",
            "fuse",
            "rerank",
            "grade_retrieval",
            "crag_rewrite_query",
            "crag_web_search",
            "build_citations",
            "choose_model",
        ]:
            pre_gen_graph.add_node(name, getattr(self, f"_node_{name}"))

        pre_gen_graph.add_edge(START, "check_cache")
        pre_gen_graph.add_conditional_edges(
            "check_cache",
            self._route_after_cache_check,
            {"cache_hit": END, "cache_miss": "detect_intent"},
        )
        pre_gen_graph.add_edge("detect_intent", "extract_keywords")
        pre_gen_graph.add_edge("extract_keywords", "determine_mode")
        pre_gen_graph.add_edge("determine_mode", "rewrite_query")
        pre_gen_graph.add_edge("rewrite_query", "build_request")
        pre_gen_graph.add_conditional_edges(
            "build_request",
            self._route_after_build_request,
            {"decompose": "decompose_query", "retrieve": "retrieve"},
        )
        pre_gen_graph.add_edge("decompose_query", "retrieve")
        pre_gen_graph.add_edge("retrieve", "graph_retrieve")
        pre_gen_graph.add_edge("graph_retrieve", "fuse")
        pre_gen_graph.add_edge("fuse", "rerank")
        pre_gen_graph.add_edge("rerank", "grade_retrieval")
        pre_gen_graph.add_conditional_edges(
            "grade_retrieval",
            self._route_after_grade_retrieval,
            {"correct": "build_citations", "needs_web_search": "crag_rewrite_query"},
        )
        pre_gen_graph.add_edge("crag_rewrite_query", "crag_web_search")
        pre_gen_graph.add_edge("crag_web_search", "build_citations")
        pre_gen_graph.add_edge("build_citations", "choose_model")
        pre_gen_graph.add_edge("choose_model", END)

        self._pre_generate_graph = pre_gen_graph.compile()

    def run(self, query: str, top_k: int, filters: dict[str, Any]) -> RagWorkflowState:
        """Execute workflow and return final state."""

        initial_state: RagWorkflowState = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        return self._graph.invoke(initial_state)

    def run_until_generate(
        self, query: str, top_k: int, filters: dict[str, Any]
    ) -> RagWorkflowState:
        """Execute pipeline up to (and including) choose_model, skipping generate_answer/store_cache."""

        initial_state: RagWorkflowState = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
        }
        return self._pre_generate_graph.invoke(initial_state)

    @property
    def answer_generator(self) -> RoutedAnswerGenerator:
        return self._answer_generator

    @property
    def query_cache(self) -> QueryCache | None:
        return self._query_cache

    def _traced_node(
        self,
        name: str,
        func: Callable[[RagWorkflowState], RagWorkflowState],
        state: RagWorkflowState,
    ) -> RagWorkflowState:
        tracing = get_tracing()
        start = time.perf_counter()
        with tracing.span(name, node=name):
            try:
                result = func(state)
                elapsed = time.perf_counter() - start
                PIPELINE_NODE_LATENCY.labels(node=name).observe(elapsed)
                return result
            except Exception:
                PIPELINE_NODE_ERRORS.labels(node=name).inc()
                raise

    def _node_detect_intent(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("detect_intent", self._impl_detect_intent, state)

    def _impl_detect_intent(self, state: RagWorkflowState) -> RagWorkflowState:
        detection = self._query_processor.detect_intent(state["query"])
        return {
            "intent": str(detection.get("intent", "factual")),
            "complexity": str(detection.get("complexity", "medium")),
        }

    def _node_extract_keywords(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("extract_keywords", self._impl_extract_keywords, state)

    def _impl_extract_keywords(self, state: RagWorkflowState) -> RagWorkflowState:
        result = self._query_processor.extract_keywords(state["query"])
        return {"hl_keywords": result.hl_keywords, "ll_keywords": result.ll_keywords}

    def _node_determine_mode(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("determine_mode", self._impl_determine_mode, state)

    def _impl_determine_mode(self, state: RagWorkflowState) -> RagWorkflowState:
        mode = self._query_processor.determine_retrieval_mode(
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            hl_keywords=state.get("hl_keywords", []),
            ll_keywords=state.get("ll_keywords", []),
        )
        return {"retrieval_mode": mode.value}

    def _node_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("rewrite_query", self._impl_rewrite_query, state)

    def _impl_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
        rewritten = self._query_processor.rewrite_query(
            query=state["query"],
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            ll_keywords=state.get("ll_keywords", []),
        )
        return {"rewritten_query": rewritten}

    def _node_build_request(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("build_request", self._impl_build_request, state)

    def _impl_build_request(self, state: RagWorkflowState) -> RagWorkflowState:
        top_k = state["top_k"]
        retrieval_query = state.get("rewritten_query") or state["query"]
        query_embedding = None
        hyde_hypothesis = None
        hyde_embeddings = None
        hyde_strategy = "disabled"

        try:
            if self._hyde_retriever and self._settings.enable_hyde:
                logger.info(
                    "HyDE build_request: invoking HyDE retriever for query=%r (len=%d tokens)",
                    retrieval_query[:80],
                    len(retrieval_query.split()),
                )
                hyde_result = self._hyde_retriever.get_query_embeddings(retrieval_query)
                hyde_embeddings = hyde_result.get("embeddings", [])
                hyde_strategy = hyde_result.get("strategy", "disabled")
                if hyde_embeddings:
                    query_embedding = hyde_embeddings[0]
                    logger.info(
                        "HyDE build_request: strategy=%s, sources=%s, embedding_count=%d, embedding_dim=%d",
                        hyde_strategy,
                        hyde_result.get("sources", []),
                        len(hyde_embeddings),
                        len(query_embedding) if query_embedding else 0,
                    )
                else:
                    logger.warning("HyDE returned no embeddings, falling back to standard embedding")
                    hyde_strategy = "fallback"
                    query_embedding = self._query_processor.build_query_embedding(retrieval_query)
            else:
                if not self._settings.enable_hyde:
                    logger.info("HyDE build_request: HyDE disabled via config")
                elif not self._hyde_retriever:
                    logger.info("HyDE build_request: HyDE retriever not initialized")
                query_embedding = self._query_processor.build_query_embedding(retrieval_query)
        except Exception as e:
            logger.exception("HyDE retrieval failed, falling back to standard embedding: %s", e)
            hyde_strategy = "fallback"
            query_embedding = self._query_processor.build_query_embedding(retrieval_query)

        request = RetrieveRequest(
            query=retrieval_query,
            query_embedding=query_embedding,
            k_sparse=max(top_k * 4, 20),
            k_dense=max(top_k * 4, 20),
            k_final=top_k,
            filters=state["filters"],
        )

        should_decompose = False
        if self._query_decomposer and self._settings.enable_query_decomposition:
            complexity = state.get("complexity", "medium")
            decision = self._query_decomposer.should_decompose(retrieval_query, complexity)
            should_decompose = decision.should_decompose
            logger.info(
                "Decomposition build_request: should_decompose=%s, confidence=%.2f, est_subq=%d, reasoning=%s",
                should_decompose, decision.confidence,
                decision.estimated_sub_questions, decision.reasoning,
            )
        else:
            logger.info(
                "Decomposition build_request: SKIPPED (enabled=%s, decomposer_init=%s)",
                self._settings.enable_query_decomposition, self._query_decomposer is not None,
            )

        return {
            "request": request,
            "query_embedding": query_embedding,
            "hyde_hypothesis": hyde_hypothesis,
            "hyde_embeddings": hyde_embeddings,
            "hyde_strategy": hyde_strategy,
            "should_decompose": should_decompose,
        }

    def _node_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("retrieve", self._impl_retrieve, state)

    def _impl_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        """Retrieve documents, handling decomposition if sub-questions exist."""
        decomposition_used = state.get("decomposition_used", False)
        sub_questions = state.get("sub_questions", [])
        request = state["request"]
        
        # Path 1: Decomposition with sub-questions
        if decomposition_used and sub_questions and self._decomposition_retriever:
            logger.info(
                "Retrieve: decomposition path — retrieving %d sub-questions in parallel",
                len(sub_questions),
            )
            try:
                # Extract question strings from SubQuestion objects
                sub_q_strings = [sq.question for sq in sub_questions]
                
                # Parallel retrieval
                sub_hits_list = self._decomposition_retriever.retrieve_sub_questions_parallel(
                    sub_q_strings,
                    request,
                    timeout_seconds=self._settings.decomposition_timeout_s,
                )
                
                # Store for potential later use (e.g., synthesis)
                state["sub_question_hits_list"] = sub_hits_list
                
                # Merge: collect all unique hits (by source_id) from all sub-questions
                merged_hits = self._merge_decomposition_hits(sub_hits_list, request.k_final)
                
                RETRIEVAL_HIT_COUNT.observe(len(merged_hits))
                logger.info(
                    "Retrieve: decomposition merge complete — %d merged hits from %d sub-questions",
                    len(merged_hits),
                    len(sub_questions),
                )
                return {
                    "hits": merged_hits,
                    "sub_question_hits_list": sub_hits_list,
                }
            except Exception as e:
                logger.exception(
                    "Retrieve: decomposition retrieval failed, falling back to main query: %s", e
                )
                # Fall through to standard retrieval
        
        # Path 2: Standard single-query retrieval
        logger.info(
            "Retrieve: standard path — query=%r (decomposition_used=%s, sub_questions=%d)",
            request.query[:60],
            decomposition_used,
            len(sub_questions),
        )
        hits = self._repository.retrieve(request)
        RETRIEVAL_HIT_COUNT.observe(len(hits))
        return {"hits": hits}

    
    def _merge_decomposition_hits(
        self,
        hits_lists: list[list[dict[str, Any]]],
        k_final: int,
        dedup_key: str = "sourceId",
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate hits from multiple sub-questions.
        
        Strategy: Collect all hits, group by dedup_key, keep first occurrence,
        maintain relative ranking by appearance order (earlier = higher confidence).
        
        Args:
            hits_lists: List of hit lists from each sub-question retrieval
            k_final: Target number of merged hits
            dedup_key: Field to use for deduplication (default: sourceId)
        
        Returns:
            Deduplicated and ranked hits list
        """
        seen_ids = set()
        merged = []
        
        for hits in hits_lists:
            for hit in hits:
                hit_id = hit.get(dedup_key)
                if hit_id and hit_id not in seen_ids:
                    merged.append(hit)
                    seen_ids.add(hit_id)
                    if len(merged) >= k_final:
                        logger.info(
                            "_merge_decomposition_hits: reached k_final=%d, stopping",
                            k_final,
                        )
                        return merged
        
        logger.info(
            "_merge_decomposition_hits: merged %d unique hits (from %d hit lists, k_final=%d)",
            len(merged),
            len(hits_lists),
            k_final,
        )
        return merged
    def _node_graph_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("graph_retrieve", self._impl_graph_retrieve, state)

    def _impl_graph_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        mode_str = state.get("retrieval_mode", "mix")
        try:
            mode = RetrievalMode(mode_str)
        except ValueError:
            mode = RetrievalMode.MIX

        if mode.skip_graph or not self._graph_retriever:
            graph_context = GraphContext()
        else:
            strategy = mode.graph_strategy or "hybrid"
            query = state.get("rewritten_query") or state["query"]

            try:
                graph_context = self._graph_retriever.retrieve(query, strategy=strategy)
            except Exception:
                logger.exception("Graph retrieval failed, falling back to empty context")
                graph_context = GraphContext()

        community_data: list[dict[str, Any]] = []
        if self._community_store and self._settings.enable_community_detection:
            try:
                summaries = self._community_store.get_community_summaries(level=0)
                top_summaries = summaries[: self._settings.community_top_k]
                community_data = [
                    {
                        "title": s.title,
                        "summary": s.summary,
                        "findings": s.findings,
                        "rating": s.rating,
                        "entity_count": s.entity_count,
                    }
                    for s in top_summaries
                ]
                logger.info(
                    "Community retrieval: %d summaries (top_k=%d)",
                    len(community_data),
                    self._settings.community_top_k,
                )
            except Exception:
                logger.exception("Community summary retrieval failed")
                community_data = []

        return {"graph_context": graph_context, "community_summaries": community_data}

    def _node_fuse(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("fuse", self._impl_fuse, state)

    def _impl_fuse(self, state: RagWorkflowState) -> RagWorkflowState:
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
        return self._traced_node("rerank", self._impl_rerank, state)

    def _impl_rerank(self, state: RagWorkflowState) -> RagWorkflowState:
        hits = state.get("fused_hits") or state.get("hits", [])
        top_k = state.get("top_k", 8)
        reranked = self._reranker.rerank(
            query=state.get("rewritten_query") or state["query"],
            hits=hits,
            top_k=top_k,
        )
        return {"reranked_hits": reranked}

    def _node_grade_retrieval(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("grade_retrieval", self._impl_grade_retrieval, state)

    def _impl_grade_retrieval(self, state: RagWorkflowState) -> RagWorkflowState:
        if not self._settings.enable_crag or not self._retrieval_grader:
            logger.info("CRAG disabled, skipping retrieval grading")
            return {"retrieval_verdict": "correct"}

        query = state.get("rewritten_query") or state.get("query", "")
        hits = state.get("reranked_hits") or state.get("fused_hits") or []
        if not hits:
            logger.info("CRAG: no hits to grade → verdict=incorrect")
            return {"retrieval_verdict": "incorrect", "reranked_hits": []}

        logger.info("CRAG grading %d hits for query=%r", len(hits), query[:80])
        relevant, verdict = self._retrieval_grader.grade_hits(query, hits)
        return {
            "retrieval_verdict": verdict,
            "reranked_hits": relevant if relevant else hits,
        }

    def _node_crag_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("crag_rewrite_query", self._impl_crag_rewrite_query, state)

    def _impl_crag_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
        query = state.get("query", "")
        if self._crag_query_rewriter:
            rewritten = self._crag_query_rewriter.rewrite(query)
        else:
            rewritten = query
        return {"crag_rewritten_query": rewritten}

    def _node_crag_web_search(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("crag_web_search", self._impl_crag_web_search, state)

    def _impl_crag_web_search(self, state: RagWorkflowState) -> RagWorkflowState:
        if not self._crag_web_searcher:
            return {}

        query = state.get("crag_rewritten_query") or state.get("query", "")
        web_hits = self._crag_web_searcher.search(query)
        if not web_hits:
            logger.info("CRAG web search returned no results")
            return {"web_search_results": []}

        verdict = state.get("retrieval_verdict", "incorrect")
        existing_hits = state.get("reranked_hits") or []

        if verdict == "ambiguous":
            merged = existing_hits + web_hits
            logger.info(
                "CRAG merge (ambiguous): %d existing + %d web = %d total",
                len(existing_hits),
                len(web_hits),
                len(merged),
            )
        else:
            merged = web_hits if web_hits else existing_hits
            logger.info(
                "CRAG replace (incorrect): %d existing → %d web hits",
                len(existing_hits),
                len(web_hits),
            )

        return {"reranked_hits": merged, "web_search_results": web_hits}

    @staticmethod
    def _route_after_grade_retrieval(state: RagWorkflowState) -> str:
        verdict = state.get("retrieval_verdict", "correct")
        logger.info("CRAG routing: verdict=%s → %s", verdict, "correct" if verdict == "correct" else "needs_web_search")
        if verdict == "correct":
            return "correct"
        return "needs_web_search"

    def _node_build_citations(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("build_citations", self._impl_build_citations, state)

    def _impl_build_citations(self, state: RagWorkflowState) -> RagWorkflowState:
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
        return self._traced_node("choose_model", self._impl_choose_model, state)

    def _impl_choose_model(self, state: RagWorkflowState) -> RagWorkflowState:
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
        return self._traced_node("generate_answer", self._impl_generate_answer, state)

    def _impl_generate_answer(self, state: RagWorkflowState) -> RagWorkflowState:
        hl = state.get("hl_keywords", [])
        ll = state.get("ll_keywords", [])
        keywords = list(dict.fromkeys(hl + ll))
        graph_context: GraphContext | None = state.get("graph_context")
        community_summaries = state.get("community_summaries")
        answer, used_model = self._answer_generator.generate(
            query=state["query"],
            hits=state.get("reranked_hits") or state.get("hits", []),
            preferred_model=state.get("preferred_model", "nova-lite"),
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            keywords=keywords or None,
            graph_context=graph_context,
            community_summaries=community_summaries,
        )
        return {"answer": answer, "answer_model": used_model}

    def _node_check_cache(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("check_cache", self._impl_check_cache, state)

    def _impl_check_cache(self, state: RagWorkflowState) -> RagWorkflowState:
        if not self._query_cache or not self._settings.enable_query_cache:
            return {"cache_hit": False}

        query = state["query"]
        try:
            embedding = self._query_processor.build_query_embedding(query)
        except Exception:
            logger.exception("Failed to build embedding for cache lookup")
            return {"cache_hit": False}

        if not embedding:
            return {"cache_hit": False}

        try:
            result = self._query_cache.lookup(embedding)
        except Exception:
            logger.exception("Cache lookup failed, continuing without cache")
            CACHE_OPERATIONS.labels(result="miss").inc()
            return {"cache_hit": False, "query_embedding": embedding}

        if result is None:
            CACHE_OPERATIONS.labels(result="miss").inc()
            return {"cache_hit": False, "query_embedding": embedding}

        CACHE_OPERATIONS.labels(result="hit").inc()
        return {
            "cache_hit": True,
            "answer": result["answer"],
            "citations": result["citations"],
            "answer_model": result["model_used"],
            "query_embedding": embedding,
        }

    @staticmethod
    def _route_after_cache_check(state: RagWorkflowState) -> str:
        if state.get("cache_hit"):
            return "cache_hit"
        return "cache_miss"

    def _node_store_cache(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("store_cache", self._impl_store_cache, state)

    def _impl_store_cache(self, state: RagWorkflowState) -> RagWorkflowState:
        if not self._query_cache or not self._settings.enable_query_cache:
            return {}

        embedding = state.get("query_embedding")
        if not embedding:
            return {}

        answer = state.get("answer", "")
        if not answer:
            return {}

        hits = state.get("reranked_hits") or state.get("hits", [])
        source_doc_ids = list({h.get("doc_id", "") for h in hits if h.get("doc_id")})

        try:
            self._query_cache.store(
                query_original=state["query"],
                query_rewritten=state.get("rewritten_query"),
                query_embedding=embedding,
                answer=answer,
                citations=state.get("citations", []),
                model_used=state.get("answer_model", ""),
                source_doc_ids=source_doc_ids,
            )
        except Exception:
            logger.exception("Failed to store result in query cache")

        return {}

    def _node_decompose_query(self, state: RagWorkflowState) -> RagWorkflowState:
        return self._traced_node("decompose_query", self._impl_decompose_query, state)

    def _impl_decompose_query(self, state: RagWorkflowState) -> RagWorkflowState:
        if not self._query_decomposer or not self._settings.enable_query_decomposition:
            logger.info("Decomposition: disabled (enable_query_decomposition=%s, decomposer=%s)",
                        self._settings.enable_query_decomposition, self._query_decomposer is not None)
            return {
                "decomposition_used": False,
                "decomposition_decision": None,
                "sub_questions": [],
                "sub_question_hits_list": [],
            }

        query = state.get("rewritten_query") or state["query"]
        logger.info("Decomposition: generating sub-questions for query=%r (len=%d tokens)",
                    query[:100], len(query.split()))

        try:
            result = self._query_decomposer.decompose_query(query)
        except Exception as e:
            logger.exception("Decomposition: LLM generation failed: %s", e)
            return {
                "decomposition_used": False,
                "decomposition_decision": None,
                "sub_questions": [],
                "sub_question_hits_list": [],
            }

        if result.should_decompose and result.sub_questions:
            sub_q_summaries = [
                f"  [{sq.id}] {sq.question[:80]} (focus={sq.focus}, strategy={sq.retrieve_strategy})"
                for sq in result.sub_questions
            ]
            logger.info(
                "Decomposition: SUCCESS — generated %d sub-questions for query=%r:\n%s",
                len(result.sub_questions),
                query[:80],
                "\n".join(sub_q_summaries),
            )
            return {
                "decomposition_decision": result,
                "sub_questions": result.sub_questions,
                "decomposition_used": True,
            }
        else:
            logger.info(
                "Decomposition: skipped after LLM call — should_decompose=%s, sub_questions=%d, reasoning=%s",
                result.should_decompose,
                len(result.sub_questions),
                result.decision_reasoning,
            )
            return {
                "decomposition_decision": result,
                "decomposition_used": False,
                "sub_questions": [],
                "sub_question_hits_list": [],
            }

    def _route_after_build_request(self, state: RagWorkflowState) -> str:
        should = state.get("should_decompose", False)
        logger.info("Decomposition routing: should_decompose=%s → %s",
                    should, "decompose" if should else "retrieve")
        if should:
            return "decompose"
        return "retrieve"

