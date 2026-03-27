"""LangGraph workflow for retrieval, routing, and grounded answer generation."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .answer_generator import ModelRoute, RoutedAnswerGenerator
from .config import Settings
from .models import RetrieveRequest
from .query_processing import QueryProcessor
from .repository import PostgresRepository


class RagWorkflowState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""

    query: str
    top_k: int
    filters: dict[str, Any]
    intent: str
    complexity: str
    rewritten_query: str
    request: RetrieveRequest
    hits: list[dict[str, Any]]
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
    ) -> None:
        """Create and compile graph with injected services."""

        # Persist service dependencies used by node handlers.
        self._settings = settings
        self._repository = repository
        self._query_processor = query_processor
        self._answer_generator = answer_generator

        # Initialize typed graph state.
        graph = StateGraph(RagWorkflowState)

        # Step 1: detect intent + complexity (prefer Qwen).
        graph.add_node("detect_intent", self._node_detect_intent)
        # Step 2: rewrite query for retrieval (prefer Qwen, fallback pass-through).
        graph.add_node("rewrite_query", self._node_rewrite_query)
        # Step 3: build retrieval request from processed query.
        graph.add_node("build_request", self._node_build_request)
        # Step 4: retrieve evidence from PostgreSQL.
        graph.add_node("retrieve", self._node_retrieve)
        # Step 5: normalize citations for action response payload.
        graph.add_node("build_citations", self._node_build_citations)
        # Step 6: choose answer model (Nova default, Qwen for complex/weak evidence).
        graph.add_node("choose_model", self._node_choose_model)
        # Step 7: generate final answer from evidence using routed model.
        graph.add_node("generate_answer", self._node_generate_answer)

        # Define deterministic execution order.
        graph.add_edge(START, "detect_intent")
        graph.add_edge("detect_intent", "rewrite_query")
        graph.add_edge("rewrite_query", "build_request")
        graph.add_edge("build_request", "retrieve")
        graph.add_edge("retrieve", "build_citations")
        graph.add_edge("build_citations", "choose_model")
        graph.add_edge("choose_model", "generate_answer")
        graph.add_edge("generate_answer", END)

        # Compile once; runtime only invokes prepared graph.
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
        """Detect query intent/complexity for rewrite and model routing."""

        detection = self._query_processor.detect_intent(state["query"])
        return {
            "intent": str(detection.get("intent", "factual")),
            "complexity": str(detection.get("complexity", "medium")),
        }

    def _node_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
        """Rewrite query to improve retrieval quality while preserving constraints."""

        rewritten = self._query_processor.rewrite_query(
            query=state["query"],
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
        )
        return {"rewritten_query": rewritten}

    def _node_build_request(self, state: RagWorkflowState) -> RagWorkflowState:
        """Build internal retrieval request payload."""

        top_k = state["top_k"]
        retrieval_query = state.get("rewritten_query") or state["query"]
        request = RetrieveRequest(
            query=retrieval_query,
            # Over-fetch sparse candidates so final top-k quality is more stable.
            k_sparse=max(top_k * 4, 20),
            k_final=top_k,
            filters=state["filters"],
        )
        return {"request": request}

    def _node_retrieve(self, state: RagWorkflowState) -> RagWorkflowState:
        """Execute retrieval against PostgreSQL repository."""

        hits = self._repository.retrieve(state["request"])
        return {"hits": hits}

    def _node_build_citations(self, state: RagWorkflowState) -> RagWorkflowState:
        """Map retrieval hits to compact citation objects for API responses."""

        citations = [
            {
                "sourceId": hit["chunk_id"],
                "title": hit["citation"]["title"],
                "url": hit["citation"]["url"],
                "snippet": hit["chunk_text"][:280],
            }
            for hit in state.get("hits", [])
        ]
        return {"citations": citations}

    def _node_choose_model(self, state: RagWorkflowState) -> RagWorkflowState:
        """Choose answer model based on complexity and retrieval confidence."""

        hits = state.get("hits", [])
        top_score = float(hits[0]["score"]) if hits else 0.0
        complexity = state.get("complexity", "medium")
        token_count = len((state.get("rewritten_query") or state["query"]).split())
        weak_evidence = len(hits) < self._settings.route_min_hits or (
            top_score < self._settings.route_top_score_threshold
        )
        complex_query = complexity == "high" or (
            token_count >= self._settings.route_complex_query_token_threshold
        )

        # Route hard/uncertain cases to Qwen Plus; keep Nova Lite as default.
        preferred_model: ModelRoute = (
            "qwen-plus" if (complex_query or weak_evidence) else "nova-lite"
        )
        return {"preferred_model": preferred_model}

    def _node_generate_answer(self, state: RagWorkflowState) -> RagWorkflowState:
        """Generate final answer using routed model selection."""

        answer, used_model = self._answer_generator.generate(
            query=state["query"],
            hits=state.get("hits", []),
            preferred_model=state.get("preferred_model", "nova-lite"),
        )
        return {"answer": answer, "answer_model": used_model}
