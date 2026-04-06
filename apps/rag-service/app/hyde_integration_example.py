"""
HyDE Integration Example for AWS Bedrock RAG Service.

Shows how to integrate HyDE into the existing retrieval pipeline.
Use this as a reference for workflow.py modifications.
"""

import logging
from typing import Optional

from .hyde_retriever import HyDEConfig, HyDERetriever
from .query_router import QueryRouter, RetrievalStrategy

logger = logging.getLogger(__name__)


def integrate_hyde_into_workflow(
    existing_workflow_state: dict,
    llm_client,
    embedding_model,
    bm25_ranker=None,
    enable_routing: bool = True,
) -> dict:
    """Example integration into existing workflow.

    This function demonstrates how to add HyDE to the current retrieval pipeline.

    Current pipeline (from workflow.py):
        detect_intent → extract_keywords → determine_mode → rewrite_query
        → build_request → retrieve → graph_retrieve → fuse → rerank
        → build_citations → choose_model → generate_answer

    HyDE integration point: After `build_request`, before `retrieve`:
        - If HyDE enabled: generate hypothesis embeddings
        - Apply query router to select strategy
        - Modify query_embedding before retrieval

    Args:
        existing_workflow_state: Current workflow state dict
        llm_client: Bedrock LLM client (e.g., ChatBedrock with Nova Pro)
        embedding_model: Embedding model instance
        bm25_ranker: Optional BM25 for semantic gap estimation
        enable_routing: Whether to use adaptive query routing

    Returns:
        Modified workflow state with HyDE query embeddings
    """

    # Initialize components
    hyde_config = HyDEConfig(
        enabled=True,
        min_query_length=5,
        temperature=0.7,
        max_hypothesis_tokens=500,
        include_original=True,  # Dual strategy
        num_hypotheses=1,  # Simple mode (set to 5 for multi-hypothesis)
    )

    hyde = HyDERetriever(llm_client, embedding_model, config=hyde_config)
    router = QueryRouter(bm25_ranker=bm25_ranker, embeddings=embedding_model)

    # Extract query from state
    query = existing_workflow_state.get("query", "")
    if not query:
        logger.warning("No query in workflow state, skipping HyDE")
        return existing_workflow_state

    # Step 1: Query analysis & routing
    if enable_routing:
        analysis = router.analyze_query(query, estimate_semantic_gap=True)
        logger.info(f"Query analysis: {analysis.strategy.value} (conf={analysis.confidence:.2f})")

        # Route decision
        if analysis.strategy == RetrievalStrategy.BM25_PRIMARY:
            logger.info("Routing to BM25-only retrieval (skip HyDE)")
            return existing_workflow_state

        elif analysis.strategy == RetrievalStrategy.HYDE_PRIMARY:
            logger.info("Routing to HyDE-primary retrieval")
            use_hyde = True

        else:  # HYBRID
            logger.info("Routing to hybrid HyDE + BM25 retrieval")
            use_hyde = True

    else:
        use_hyde = True

    # Step 2: Generate HyDE embeddings
    if use_hyde:
        try:
            embeddings_result = hyde.get_query_embeddings(query)
            logger.info(
                f"HyDE strategy: {embeddings_result['strategy']} | "
                f"sources: {embeddings_result['sources']}"
            )

            # Modify workflow state
            # Replace or augment the existing query_embedding
            existing_workflow_state["query_embeddings_hyde"] = embeddings_result["embeddings"]
            existing_workflow_state["hyde_sources"] = embeddings_result["sources"]
            existing_workflow_state["hyde_strategy"] = embeddings_result["strategy"]
            existing_workflow_state["use_hyde"] = True

        except Exception as e:
            logger.error(f"HyDE generation failed, falling back to original: {e}")
            existing_workflow_state["use_hyde"] = False

    return existing_workflow_state


def hybrid_retrieve_with_hyde(
    state: dict,
    repository,  # OpenSearch + pgvector repository
    k_sparse: int = 20,
    k_final: int = 5,
    weights_rrf: Optional[dict] = None,
) -> dict:
    """Example hybrid retrieval with HyDE embeddings.

    Modifies retrieve node to use HyDE embeddings when available.

    Args:
        state: Workflow state with HyDE embeddings
        repository: Retrieval repository (OpenSearch + pgvector)
        k_sparse: Number of sparse candidates to retrieve
        k_final: Final top-k results
        weights_rrf: RRF weights for fusion (sparse_weight, dense_weight)

    Returns:
        Modified state with retrieved documents
    """

    query = state.get("query", "")
    use_hyde = state.get("use_hyde", False)
    query_embeddings = state.get("query_embeddings_hyde")

    if weights_rrf is None:
        weights_rrf = {"sparse": 0.4, "dense": 0.6}

    # Retrieve with HyDE or original embedding
    if use_hyde and query_embeddings:
        logger.info(f"Retrieving with HyDE embeddings ({len(query_embeddings)} embedding(s))")

        # If multiple embeddings (e.g., hypothesis + original), retrieve with each
        all_dense_results = []
        for i, emb in enumerate(query_embeddings):
            try:
                # Sparse retrieval (standard BM25)
                sparse_results = repository.search_sparse(query, k=k_sparse)

                # Dense retrieval with HyDE embedding
                dense_results = repository.search_dense(embedding=emb, k=k_sparse)

                # Fuse with RRF
                fused = repository.fuse_results(
                    sparse_results=sparse_results,
                    dense_results=dense_results,
                    k=k_final,
                    weights_sparse=weights_rrf["sparse"],
                    weights_dense=weights_rrf["dense"],
                    method="rrf",
                )

                all_dense_results.extend(fused)
                logger.debug(f"Embedding {i+1}: retrieved {len(fused)} results")

            except Exception as e:
                logger.error(f"Dense retrieval with embedding {i} failed: {e}")

        # Deduplicate results if multiple embeddings were used
        if len(query_embeddings) > 1:
            seen_ids = set()
            deduped = []
            for result in all_dense_results:
                if result["id"] not in seen_ids:
                    seen_ids.add(result["id"])
                    deduped.append(result)
            state["documents"] = deduped[:k_final]
            logger.info(f"Deduped results: {len(deduped[:k_final])} / {len(all_dense_results)}")
        else:
            state["documents"] = all_dense_results[:k_final]

    else:
        # Fallback: standard retrieval
        logger.info("Retrieving with original query embedding (HyDE disabled)")
        sparse_results = repository.search_sparse(query, k=k_sparse)
        dense_results = repository.search_dense(query_embedding=state.get("query_embedding"), k=k_sparse)

        fused = repository.fuse_results(
            sparse_results=sparse_results,
            dense_results=dense_results,
            k=k_final,
            weights_sparse=weights_rrf["sparse"],
            weights_dense=weights_rrf["dense"],
            method="rrf",
        )

        state["documents"] = fused

    return state


def hyde_cost_analysis(
    num_queries_per_month: int = 1_000_000,
    bedrock_nova_pro_rate: float = 0.80,  # $/1M tokens
    hypothesis_tokens: int = 200,
    hyde_percentage: float = 0.4,  # 40% of queries use HyDE
    rerank_percentage: float = 0.1,  # 10% of queries reranked
    rerank_tokens: int = 1500,
    rerank_rate: float = 0.60,  # Claude Sonnet rate $/1M
) -> dict:
    """Calculate monthly HyDE costs.

    Args:
        num_queries_per_month: Total queries
        bedrock_nova_pro_rate: Nova Pro cost per 1M tokens
        hypothesis_tokens: Avg tokens per hypothesis
        hyde_percentage: % of queries using HyDE
        rerank_percentage: % of queries reranked
        rerank_tokens: Avg tokens for reranking
        rerank_rate: Rerank model cost per 1M tokens

    Returns:
        Cost analysis dict
    """
    hyde_queries = num_queries_per_month * hyde_percentage
    hyde_tokens = hyde_queries * hypothesis_tokens
    hyde_cost = (hyde_tokens / 1_000_000) * bedrock_nova_pro_rate

    rerank_queries = num_queries_per_month * rerank_percentage
    rerank_total_tokens = rerank_queries * rerank_tokens
    rerank_cost = (rerank_total_tokens / 1_000_000) * rerank_rate

    total_cost = hyde_cost + rerank_cost

    return {
        "hyde_queries": int(hyde_queries),
        "hyde_tokens": int(hyde_tokens),
        "hyde_cost": round(hyde_cost, 2),
        "rerank_queries": int(rerank_queries),
        "rerank_tokens": int(rerank_total_tokens),
        "rerank_cost": round(rerank_cost, 2),
        "total_cost_per_month": round(total_cost, 2),
        "optimization_tips": [
            "Enable HyDE for long queries only (>15 tokens) → -40% cost",
            "Sample 10% of queries for reranking → -90% rerank cost",
            "Cache popular query results → -40-70% cost",
            "Use Nova 2 Lite for hypothesis generation → -50% cost (but test quality)",
        ],
    }


if __name__ == "__main__":
    # Example: show cost analysis
    costs = hyde_cost_analysis()
    print("\n=== HyDE Monthly Cost Analysis ===")
    print(f"HyDE queries: {costs['hyde_queries']:,}")
    print(f"HyDE cost: ${costs['hyde_cost']:.2f}")
    print(f"Reranking queries: {costs['rerank_queries']:,}")
    print(f"Reranking cost: ${costs['rerank_cost']:.2f}")
    print(f"Total monthly cost: ${costs['total_cost_per_month']:.2f}")
    print("\nOptimization tips:")
    for tip in costs["optimization_tips"]:
        print(f"  • {tip}")
