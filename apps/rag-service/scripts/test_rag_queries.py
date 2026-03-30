"""Test RAG queries with/without entity-enhanced graph retrieval.

Usage:
    cd apps/rag-service && source ../../.envrc && export QWEN_API_KEY=...
    python -m scripts.test_rag_queries                              # graph-enhanced
    python -m scripts.test_rag_queries --compare                    # graph vs chunks-only
    python -m scripts.test_rag_queries --query "Who wrote Rushing Back?"
    python -m scripts.test_rag_queries --no-answer                  # retrieval only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.answer_generator import (
    BedrockConverseAnswerGenerator,
    QwenAnswerGenerator,
    RoutedAnswerGenerator,
)
from app.config import Settings, get_settings
from app.entity_vector_store import EntityVectorStore
from app.graph_retriever import GraphRetriever
from app.models import GraphContext, RetrievalMode
from app.query_processing import QueryProcessor
from app.qwen_client import QwenClient
from app.repository import PostgresRepository
from app.reranker import LLMReranker
from app.secrets import resolve_neo4j_password
from app.tracing import init_tracing
from app.workflow import RagWorkflow, RagWorkflowState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_rag_queries")

for name in ("httpx", "httpcore", "urllib3", "botocore", "boto3", "openai"):
    logging.getLogger(name).setLevel(logging.WARNING)


TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query": "Who wrote the song Rushing Back?",
        "expected_entities": ["Rushing Back", "Flume", "Vera Blue"],
        "category": "entity_lookup",
    },
    {
        "query": "What is APRA AMCOS and what do they do?",
        "expected_entities": ["APRA AMCOS"],
        "category": "entity_description",
    },
    {
        "query": "Which organisations are involved in music licensing in Australia?",
        "expected_entities": ["APRA AMCOS", "ARIA", "AMPAL"],
        "category": "multi_entity",
    },
    {
        "query": "What territories does AMPAL operate in?",
        "expected_entities": ["AMPAL", "Australia"],
        "category": "relationship",
    },
    {
        "query": "What is the role of the Australian Publishers Association?",
        "expected_entities": ["Australian Publishers Association"],
        "category": "entity_description",
    },
    {
        "query": "What license terms apply to music performed in public venues?",
        "expected_entities": [],
        "category": "concept_query",
    },
    {
        "query": "List the identifiers used to track music works",
        "expected_entities": [],
        "category": "concept_query",
    },
    {
        "query": "What is the relationship between ARIA and PPCA?",
        "expected_entities": ["ARIA", "PPCA"],
        "category": "relationship",
    },
]


def _force_mode_patch(workflow: RagWorkflow, mode: RetrievalMode) -> None:
    """Monkey-patch workflow to always return a fixed retrieval mode."""
    workflow._impl_determine_mode = lambda state: {"retrieval_mode": mode.value}  # type: ignore[attr-defined]


def build_workflow(
    settings: Settings,
    *,
    enable_graph: bool = True,
) -> RagWorkflow:
    repository = PostgresRepository(settings)
    qwen_client = QwenClient(settings)
    query_processor = QueryProcessor(settings=settings, qwen_client=qwen_client)
    answer_generator = RoutedAnswerGenerator(
        bedrock_generator=BedrockConverseAnswerGenerator(settings),
        qwen_generator=QwenAnswerGenerator(qwen_client, settings),
    )
    reranker = LLMReranker(settings=settings, qwen_client=qwen_client)

    graph_retriever = None
    if enable_graph and settings.enable_graph_retrieval:
        vector_store = EntityVectorStore(settings)
        neo4j_repo = None
        if settings.enable_neo4j:
            from app.graph_repository import Neo4jRepository

            neo4j_password = resolve_neo4j_password(settings)
            neo4j_repo = Neo4jRepository(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=neo4j_password,
                database=settings.neo4j_database,
            )
        graph_retriever = GraphRetriever(
            qwen_client=qwen_client,
            vector_store=vector_store,
            neo4j_repo=neo4j_repo,
            settings=settings,
        )

    return RagWorkflow(
        settings=settings,
        repository=repository,
        query_processor=query_processor,
        answer_generator=answer_generator,
        reranker=reranker,
        graph_retriever=graph_retriever,
        query_cache=None,
    )


def format_graph_context(gc: GraphContext | None) -> dict[str, Any]:
    if gc is None or gc.is_empty:
        return {"entities": [], "relations": [], "source_chunk_ids": []}
    return {
        "entities": [
            {
                "name": e.name,
                "type": e.type,
                "score": round(e.score, 4),
                "description": e.description[:120],
            }
            for e in sorted(gc.entities, key=lambda x: x.score, reverse=True)
        ],
        "relations": [
            {
                "source": r.source_entity,
                "type": r.relation_type,
                "target": r.target_entity,
                "score": round(r.score, 4),
                "evidence": r.evidence[:120] if r.evidence else "",
            }
            for r in sorted(gc.relations, key=lambda x: x.score, reverse=True)
        ],
        "source_chunk_ids": gc.source_chunk_ids[:5],
    }


def format_citations(citations: list[dict[str, Any]]) -> list[str]:
    return [c.get("title", "untitled") for c in citations[:5]]


def check_expected_entities(gc: GraphContext | None, expected: list[str]) -> dict[str, bool]:
    if not expected:
        return {}
    found_names = set()
    if gc and gc.entities:
        found_names = {e.name.lower() for e in gc.entities}
    return {name: name.lower() in found_names for name in expected}


def run_query(
    workflow: RagWorkflow,
    query: str,
    *,
    top_k: int = 8,
    generate_answer: bool = True,
) -> dict[str, Any]:
    t0 = time.time()

    if generate_answer:
        state: RagWorkflowState = workflow.run(query, top_k=top_k, filters={})
    else:
        state = workflow.run_until_generate(query, top_k=top_k, filters={})

    elapsed = time.time() - t0

    gc: GraphContext | None = state.get("graph_context")  # type: ignore[assignment]
    result: dict[str, Any] = {
        "query": query,
        "elapsed_s": round(elapsed, 2),
        "intent": state.get("intent", ""),
        "complexity": state.get("complexity", ""),
        "retrieval_mode": state.get("retrieval_mode", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "num_hits": len(state.get("hits", [])),
        "num_fused_hits": len(state.get("fused_hits", [])),
        "num_reranked_hits": len(state.get("reranked_hits", [])),
        "graph_context": format_graph_context(gc),
        "citations": format_citations(state.get("citations", [])),
        "answer_model": str(state.get("answer_model", "")),
    }
    if generate_answer:
        answer = state.get("answer", "")
        result["answer"] = answer[:500] + ("..." if len(answer) > 500 else "")

    return result


def print_result(result: dict[str, Any], expected_entities: list[str] | None = None) -> None:
    print("\n" + "=" * 70)
    print(f"  QUERY: {result['query']}")
    print("=" * 70)
    print(
        f"  Intent: {result['intent']}  |  Complexity: {result['complexity']}  |  Mode: {result['retrieval_mode']}"
    )
    print(f"  Rewritten: {result['rewritten_query']}")
    print(
        f"  Hits: {result['num_hits']} raw → {result['num_fused_hits']} fused → {result['num_reranked_hits']} reranked"
    )
    print(f"  Time: {result['elapsed_s']}s  |  Model: {result['answer_model']}")

    gc = result["graph_context"]
    if gc["entities"]:
        print(f"\n  📊 Graph Entities ({len(gc['entities'])}):")
        for e in gc["entities"][:8]:
            print(f"    • {e['name']} ({e['type']}) — score={e['score']}  {e['description'][:80]}")
    else:
        print("\n  📊 Graph Entities: (none)")

    if gc["relations"]:
        print(f"\n  🔗 Graph Relations ({len(gc['relations'])}):")
        for r in gc["relations"][:5]:
            print(f"    • {r['source']} --[{r['type']}]--> {r['target']}  score={r['score']}")
    else:
        print("  🔗 Graph Relations: (none)")

    if expected_entities:
        if gc["entities"]:
            found_names = {e["name"].lower() for e in gc["entities"]}
            hits = {name: name.lower() in found_names for name in expected_entities}
        else:
            hits = {name: False for name in expected_entities}
        print("\n  ✅ Expected Entity Check:")
        for name, found in hits.items():
            status = "✅" if found else "❌"
            print(f"    {status} {name}")

    if result.get("citations"):
        print(f"\n  📄 Citations: {result['citations']}")

    if result.get("answer"):
        print(f"\n  💬 Answer:\n  {result['answer']}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RAG queries with entity graph retrieval")
    parser.add_argument("--query", "-q", type=str, help="Run a single ad-hoc query")
    parser.add_argument("--compare", action="store_true", help="Compare graph vs chunks-only")
    parser.add_argument(
        "--no-answer", action="store_true", help="Skip answer generation (retrieval only)"
    )
    parser.add_argument("--top-k", type=int, default=8, help="Top K results (default: 8)")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["local", "global", "hybrid", "mix", "graph_only", "naive"],
        help="Force retrieval mode (bypasses dynamic routing)",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Initialize OpenTelemetry tracing (same as FastAPI startup in main.py)
    tracing_svc = init_tracing(settings)
    if settings.enable_tracing:
        logger.info(f"Tracing enabled → {settings.tracing_provider} @ {settings.tracing_endpoint}")

    if not settings.enable_graph_retrieval:
        logger.warning("RAG_ENABLE_GRAPH_RETRIEVAL is not set. Setting it to True for testing.")
        os.environ["RAG_ENABLE_GRAPH_RETRIEVAL"] = "true"
        settings = get_settings()

    generate_answer = not args.no_answer

    queries: list[dict[str, Any]]
    if args.query:
        queries = [{"query": args.query, "expected_entities": [], "category": "adhoc"}]
    else:
        queries = TEST_QUERIES

    if args.compare:
        logger.info("Building graph-enhanced workflow...")
        wf_graph = build_workflow(settings, enable_graph=True)
        if args.mode:
            forced = RetrievalMode(args.mode)
            _force_mode_patch(wf_graph, forced)
            logger.info(f"Forced retrieval mode: {forced.value}")

        logger.info("Building chunks-only workflow...")
        wf_chunks = build_workflow(settings, enable_graph=False)

        all_results = []
        for q in queries:
            print(f"\n{'#' * 70}")
            print(f"  COMPARING: {q['query']}")
            print(f"{'#' * 70}")

            print("\n>>> WITH GRAPH RETRIEVAL:")
            r_graph = run_query(
                wf_graph, q["query"], top_k=args.top_k, generate_answer=generate_answer
            )
            print_result(r_graph, q.get("expected_entities"))

            print("\n>>> WITHOUT GRAPH (chunks only):")
            r_chunks = run_query(
                wf_chunks, q["query"], top_k=args.top_k, generate_answer=generate_answer
            )
            print_result(r_chunks, q.get("expected_entities"))

            all_results.append(
                {
                    "query": q["query"],
                    "category": q.get("category", ""),
                    "graph": r_graph,
                    "chunks_only": r_chunks,
                }
            )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

    else:
        logger.info("Building graph-enhanced workflow...")
        wf = build_workflow(settings, enable_graph=True)
        if args.mode:
            forced = RetrievalMode(args.mode)
            _force_mode_patch(wf, forced)
            logger.info(f"Forced retrieval mode: {forced.value}")

        all_results = []
        for q in queries:
            result = run_query(wf, q["query"], top_k=args.top_k, generate_answer=generate_answer)
            print_result(result, q.get("expected_entities"))
            all_results.append(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    total = len(queries)
    if not args.compare:
        graph_hits = sum(1 for r in all_results if r["graph_context"]["entities"])
        print(f"  Queries: {total}")
        print(f"  With graph entities: {graph_hits}/{total}")
        avg_time = sum(r["elapsed_s"] for r in all_results) / total if total else 0
        print(f"  Avg time: {avg_time:.2f}s")
    else:
        print(f"  Queries compared: {total}")
        for r in all_results:
            g_ents = len(r["graph"]["graph_context"]["entities"])
            c_ents = len(r["chunks_only"]["graph_context"]["entities"])
            print(f"  • {r['query'][:50]}... → graph:{g_ents} ents, chunks:{c_ents} ents")
    print()

    tracing_svc.shutdown()


if __name__ == "__main__":
    main()
