"""Comprehensive eval runner: golden dataset → RagWorkflow → RAGAS-compatible output.

Usage:
    cd apps/rag-service && source ../../.envrc && export QWEN_API_KEY=...
    python -m scripts.eval_runner                         # full eval
    python -m scripts.eval_runner --limit 5               # quick test
    python -m scripts.eval_runner --intent factual         # factual only
    python -m scripts.eval_runner --ragas-output /tmp/ragas_input.json  # RAGAS format
    python -m scripts.eval_runner --compare                # graph vs chunks-only
    python -m scripts.eval_runner --markdown /tmp/eval_report.md  # markdown report
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import Settings, get_settings
from app.models import GraphContext, RetrievalMode
from app.tracing import init_tracing
from app.workflow import RagWorkflow, RagWorkflowState
from scripts.test_rag_queries import _force_mode_patch, build_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_runner")

for name in ("httpx", "httpcore", "urllib3", "botocore", "boto3", "openai"):
    logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_golden_dataset(path: str) -> list[dict[str, Any]]:
    """Load and validate the golden dataset JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # v2 format with top-level "entries" key
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]

    raise ValueError(f"Unexpected golden dataset format in {path!r}")


# ---------------------------------------------------------------------------
# Entity recall computation
# ---------------------------------------------------------------------------


def _normalize_entity(name: str) -> str:
    """Normalize entity name for comparison: lowercase + underscores → spaces."""
    return name.lower().replace("_", " ").strip()


def compute_entity_recall(
    expected_entities: list[str],
    graph_context: GraphContext | None,
) -> dict[str, Any]:
    """Compute entity recall against graph context entity names.

    Returns a dict with:
        entity_recall   — float [0, 1] or None if no expected entities
        found_entities  — list of expected entities that were found
        missing_entities — list of expected entities not found
    """
    if not expected_entities:
        return {
            "entity_recall": None,
            "found_entities": [],
            "missing_entities": [],
        }

    # Build lookup from graph entities (all normalized)
    graph_entity_names: set[str] = set()
    if graph_context and graph_context.entities:
        for e in graph_context.entities:
            graph_entity_names.add(_normalize_entity(e.name))

    found: list[str] = []
    missing: list[str] = []

    for expected in expected_entities:
        normalized = _normalize_entity(expected)
        # Partial match: any graph entity contains the normalized expected name
        if any(normalized in ge or ge in normalized for ge in graph_entity_names):
            found.append(expected)
        else:
            missing.append(expected)

    recall = len(found) / len(expected_entities) if expected_entities else None

    return {
        "entity_recall": round(recall, 4) if recall is not None else None,
        "found_entities": found,
        "missing_entities": missing,
    }


# ---------------------------------------------------------------------------
# Single query runner
# ---------------------------------------------------------------------------


def run_single(
    workflow: RagWorkflow,
    entry: dict[str, Any],
    *,
    top_k: int = 8,
    generate_answer: bool = True,
) -> dict[str, Any]:
    """Run one golden entry through the RagWorkflow and return a result dict."""
    query: str = entry["query"]
    expected_entities: list[str] = entry.get("expected_entities", [])

    t0 = time.time()
    error: str | None = None
    state: RagWorkflowState = {}

    try:
        if generate_answer:
            state = workflow.run(query, top_k=top_k, filters={})
        else:
            state = workflow.run_until_generate(query, top_k=top_k, filters={})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("Query failed [%s]: %s", entry.get("id", "?"), error)

    elapsed = round(time.time() - t0, 2)

    gc: GraphContext | None = state.get("graph_context")  # type: ignore[assignment]
    recall_info = compute_entity_recall(expected_entities, gc)

    # Extract retrieved contexts from reranked_hits
    reranked_hits: list[dict[str, Any]] = state.get("reranked_hits", []) or []
    retrieved_contexts: list[str] = [
        hit.get("chunk_text", hit.get("snippet", hit.get("text", ""))) for hit in reranked_hits
    ]

    citations: list[dict[str, Any]] = state.get("citations", []) or []
    answer: str = state.get("answer", "") or ""

    result: dict[str, Any] = {
        "id": entry.get("id", ""),
        "query": query,
        "expected_answer": entry.get("expected_answer", ""),
        "actual_answer": answer,
        "intent": state.get("intent", entry.get("intent", "")),
        "complexity": state.get("complexity", entry.get("complexity", "")),
        "elapsed_s": elapsed,
        "retrieval_mode": state.get("retrieval_mode", ""),
        # Entity recall
        "entity_recall": recall_info["entity_recall"],
        "expected_entities": expected_entities,
        "found_entities": recall_info["found_entities"],
        "missing_entities": recall_info["missing_entities"],
        # Retrieval stats
        "num_citations": len(citations),
        "num_retrieved_contexts": len(retrieved_contexts),
        "retrieved_contexts": retrieved_contexts,
        # Graph stats
        "graph_entities_count": len(gc.entities) if gc else 0,
        "graph_relations_count": len(gc.relations) if gc else 0,
        # Additional metadata
        "tags": entry.get("tags", []),
        "rewritten_query": state.get("rewritten_query", ""),
        "answer_model": str(state.get("answer_model", "")),
        "num_hits": len(state.get("hits", []) or []),
        "num_fused_hits": len(state.get("fused_hits", []) or []),
        "num_reranked_hits": len(reranked_hits),
    }

    if error:
        result["error"] = error

    return result


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _recall_values(results: list[dict[str, Any]]) -> list[float]:
    return [r["entity_recall"] for r in results if r.get("entity_recall") is not None]


def compute_summary(
    results: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
    graph_enabled: bool,
    limit: int | None,
) -> dict[str, Any]:
    """Build the top-level summary block."""
    total = len(results)

    elapsed_vals = [r["elapsed_s"] for r in results]
    recall_vals = _recall_values(results)
    citation_vals = [r["num_citations"] for r in results]

    # By intent
    by_intent: dict[str, dict[str, Any]] = defaultdict(lambda: {"entries": []})
    for r in results:
        intent = r.get("intent") or "unknown"
        by_intent[intent]["entries"].append(r)

    by_intent_summary: dict[str, dict[str, Any]] = {}
    for intent, data in sorted(by_intent.items()):
        entries = data["entries"]
        by_intent_summary[intent] = {
            "count": len(entries),
            "entity_recall": _avg(_recall_values(entries)),
            "avg_elapsed_s": _avg([e["elapsed_s"] for e in entries]),
        }

    # By complexity
    by_complexity: dict[str, dict[str, Any]] = defaultdict(lambda: {"entries": []})
    for r in results:
        complexity = r.get("complexity") or "unknown"
        by_complexity[complexity]["entries"].append(r)

    by_complexity_summary: dict[str, dict[str, Any]] = {}
    for complexity, data in sorted(by_complexity.items()):
        entries = data["entries"]
        by_complexity_summary[complexity] = {
            "count": len(entries),
            "entity_recall": _avg(_recall_values(entries)),
            "avg_elapsed_s": _avg([e["elapsed_s"] for e in entries]),
        }

    return {
        "total_queries": total,
        "avg_elapsed_s": _avg(elapsed_vals),
        "entity_recall": _avg(recall_vals),
        "avg_num_citations": _avg(citation_vals),
        "by_intent": by_intent_summary,
        "by_complexity": by_complexity_summary,
    }


# ---------------------------------------------------------------------------
# RAGAS output format
# ---------------------------------------------------------------------------


def to_ragas_row(result: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Convert a result + entry into a RAGAS-compatible row."""
    return {
        "user_input": result["query"],
        "response": result["actual_answer"],
        "retrieved_contexts": result["retrieved_contexts"],
        "reference": result["expected_answer"],
        "category": entry.get("intent", ""),
    }


# ---------------------------------------------------------------------------
# Scorecard printer
# ---------------------------------------------------------------------------


THRESHOLDS: dict[str, float] = {
    "entity_recall": 0.50,
    "avg_elapsed_s": 30.0,
}


def check_thresholds(
    summary: dict[str, Any],
    *,
    custom_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Check summary metrics against thresholds and return verdicts.

    Returns a list of dicts:
        {"metric": str, "value": float, "threshold": float, "passed": bool}
    """
    thresholds = {**THRESHOLDS, **(custom_thresholds or {})}
    verdicts: list[dict[str, Any]] = []

    recall = summary.get("entity_recall", 0.0)
    verdicts.append(
        {
            "metric": "entity_recall",
            "value": recall,
            "threshold": thresholds["entity_recall"],
            "passed": recall >= thresholds["entity_recall"],
        }
    )

    avg_time = summary.get("avg_elapsed_s", 0.0)
    verdicts.append(
        {
            "metric": "avg_elapsed_s",
            "value": avg_time,
            "threshold": thresholds["avg_elapsed_s"],
            "passed": avg_time <= thresholds["avg_elapsed_s"],
        }
    )

    return verdicts


def print_scorecard(
    summary: dict[str, Any],
    *,
    run_id: str,
    timestamp: str,
    mode: str,
    top_k: int,
    verdicts: list[dict[str, Any]] | None = None,
) -> None:
    """Print a formatted scorecard to stdout."""
    total = summary["total_queries"]
    avg_time = summary["avg_elapsed_s"]
    recall = summary["entity_recall"]
    recall_pct = f"{recall * 100:.1f}%" if recall else "N/A"

    width = 56

    def row(text: str) -> str:
        padding = width - len(text) - 2
        return f"║  {text}{' ' * max(0, padding)}║"

    def divider() -> str:
        return "╠" + "═" * (width) + "╣"

    lines = [
        "╔" + "═" * width + "╗",
        row(f"RAG Eval Scorecard — {timestamp}"),
        divider(),
        row(f"Total queries: {total}  |  Mode: {mode}  |  Top-K: {top_k}"),
        row(f"Avg time: {avg_time}s    |  Entity recall: {recall_pct}"),
        divider(),
        row("By Intent:"),
    ]

    for intent, stats in sorted(summary.get("by_intent", {}).items()):
        count = stats["count"]
        r = stats["entity_recall"]
        t = stats["avg_elapsed_s"]
        r_str = f"{r * 100:.1f}%" if r else "N/A"
        lines.append(row(f"  {intent:<12} ({count:>2}): recall={r_str}  time={t}s"))

    lines.append(divider())
    lines.append(row("By Complexity:"))

    for complexity, stats in sorted(summary.get("by_complexity", {}).items()):
        count = stats["count"]
        r = stats["entity_recall"]
        t = stats["avg_elapsed_s"]
        r_str = f"{r * 100:.1f}%" if r else "N/A"
        lines.append(row(f"  {complexity:<10} ({count:>2}): recall={r_str}  time={t}s"))

    if verdicts:
        lines.append(divider())
        all_pass = all(v["passed"] for v in verdicts)
        lines.append(row(f"Threshold Gates: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}"))
        for v in verdicts:
            icon = "✅" if v["passed"] else "❌"
            lines.append(
                row(f"  {icon} {v['metric']}: {v['value']:.4f}  (threshold: {v['threshold']})")
            )

    lines.append("╚" + "═" * width + "╝")

    print("\n" + "\n".join(lines) + "\n")


def write_markdown_report(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    run_id: str,
    timestamp: str,
    mode: str,
    top_k: int,
    verdicts: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write a markdown evaluation report to disk."""
    all_pass = all(v["passed"] for v in verdicts)
    status = "✅ PASS" if all_pass else "❌ FAIL"

    lines: list[str] = [
        f"# RAG Eval Report — {timestamp}",
        "",
        f"**Run ID**: `{run_id}`  ",
        f"**Mode**: `{mode}`  |  **Top-K**: `{top_k}`  |  **Status**: {status}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total queries | {summary['total_queries']} |",
        f"| Avg elapsed (s) | {summary['avg_elapsed_s']:.2f} |",
        f"| Entity recall | {summary['entity_recall']:.4f} |",
        f"| Avg citations | {summary['avg_num_citations']:.1f} |",
        "",
        "## Threshold Gates",
        "",
        "| Metric | Value | Threshold | Status |",
        "|--------|-------|-----------|--------|",
    ]

    for v in verdicts:
        icon = "✅" if v["passed"] else "❌"
        lines.append(f"| {v['metric']} | {v['value']:.4f} | {v['threshold']} | {icon} |")

    lines.extend(
        [
            "",
            "## By Intent",
            "",
            "| Intent | Count | Entity Recall | Avg Time (s) |",
            "|--------|-------|---------------|--------------|",
        ]
    )

    for intent, stats in sorted(summary.get("by_intent", {}).items()):
        r = stats["entity_recall"]
        lines.append(f"| {intent} | {stats['count']} | {r:.4f} | {stats['avg_elapsed_s']:.2f} |")

    lines.extend(
        [
            "",
            "## By Complexity",
            "",
            "| Complexity | Count | Entity Recall | Avg Time (s) |",
            "|------------|-------|---------------|--------------|",
        ]
    )

    for complexity, stats in sorted(summary.get("by_complexity", {}).items()):
        r = stats["entity_recall"]
        lines.append(
            f"| {complexity} | {stats['count']} | {r:.4f} | {stats['avg_elapsed_s']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Query Results",
            "",
            "| ID | Query | Recall | Elapsed (s) | Entities | Error |",
            "|----|-------|--------|-------------|----------|-------|",
        ]
    )

    for r in results:
        recall_val = r.get("entity_recall")
        recall_str = f"{recall_val:.2f}" if recall_val is not None else "N/A"
        error_str = r.get("error", "")
        query_short = r["query"][:50] + ("..." if len(r["query"]) > 50 else "")
        lines.append(
            f"| {r['id']} | {query_short} | {recall_str} | {r['elapsed_s']:.1f} "
            f"| {r['graph_entities_count']} | {error_str} |"
        )

    lines.extend(["", "---", f"*Generated by eval_runner.py on {timestamp}*", ""])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Markdown report saved → %s", output_path)


# ---------------------------------------------------------------------------
# Compare mode: graph vs chunks-only
# ---------------------------------------------------------------------------


def run_compare(
    entries: list[dict[str, Any]],
    settings: Settings,
    *,
    top_k: int,
    generate_answer: bool,
    forced_mode: RetrievalMode | None,
) -> list[dict[str, Any]]:
    """Run each entry against both graph-enabled and chunks-only workflows."""
    logger.info("Building graph-enhanced workflow...")
    wf_graph = build_workflow(settings, enable_graph=True)
    if forced_mode is not None:
        _force_mode_patch(wf_graph, forced_mode)
        logger.info("Forced retrieval mode: %s", forced_mode.value)

    logger.info("Building chunks-only workflow...")
    wf_chunks = build_workflow(settings, enable_graph=False)

    compare_results: list[dict[str, Any]] = []
    total = len(entries)

    for i, entry in enumerate(entries, 1):
        logger.info("[%d/%d] Comparing: %s", i, total, entry.get("query", "")[:80])

        r_graph = run_single(wf_graph, entry, top_k=top_k, generate_answer=generate_answer)
        r_chunks = run_single(wf_chunks, entry, top_k=top_k, generate_answer=generate_answer)

        compare_results.append(
            {
                "id": entry.get("id", ""),
                "query": entry["query"],
                "intent": entry.get("intent", ""),
                "complexity": entry.get("complexity", ""),
                "graph": r_graph,
                "chunks_only": r_chunks,
            }
        )

        logger.info(
            "  → graph recall=%.2f  chunks recall=%.2f",
            r_graph.get("entity_recall") or 0.0,
            r_chunks.get("entity_recall") or 0.0,
        )

    return compare_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run golden dataset through RagWorkflow and produce eval output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="tests/eval/golden_dataset.json",
        help="Golden dataset JSON (default: tests/eval/golden_dataset.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/tmp/eval_results.json",
        help="Output file for full results (default: /tmp/eval_results.json)",
    )
    parser.add_argument(
        "--ragas-output",
        default=None,
        help="Optional RAGAS-format output path for scripts/ragas_eval.py",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["local", "global", "hybrid", "mix", "naive"],
        help="Force retrieval mode (bypasses dynamic routing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only run first N entries (for quick testing)",
    )
    parser.add_argument(
        "--intent",
        default=None,
        choices=["factual", "comparative", "analytical", "exploratory"],
        help="Filter to specific intent",
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Skip answer generation (retrieval only)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both graph-enhanced and chunks-only workflows",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top K results (default: 8)",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        metavar="PATH",
        help="Save markdown report to PATH (e.g. /tmp/eval_report.md)",
    )

    args = parser.parse_args()

    # ---- Settings & tracing ----
    settings = get_settings()
    tracing_svc = init_tracing(settings)
    if settings.enable_tracing:
        logger.info(
            "Tracing enabled → %s @ %s", settings.tracing_provider, settings.tracing_endpoint
        )

    if not settings.enable_graph_retrieval:
        logger.warning("RAG_ENABLE_GRAPH_RETRIEVAL is not set. Setting it to True for testing.")
        os.environ["RAG_ENABLE_GRAPH_RETRIEVAL"] = "true"
        settings = get_settings()

    # ---- Load dataset ----
    dataset_path = args.dataset
    logger.info("Loading golden dataset from %s", dataset_path)
    entries = load_golden_dataset(dataset_path)
    total_entries = len(entries)
    logger.info("Loaded %d entries", total_entries)

    # ---- Filter by intent ----
    if args.intent:
        entries = [e for e in entries if e.get("intent") == args.intent]
        logger.info("Filtered to intent=%r → %d entries", args.intent, len(entries))

    # ---- Apply limit ----
    if args.limit is not None:
        entries = entries[: args.limit]
        logger.info("Limited to first %d entries", len(entries))

    if not entries:
        logger.warning("No entries to evaluate — exiting.")
        tracing_svc.shutdown()
        return

    generate_answer = not args.no_answer
    top_k: int = args.top_k
    forced_mode: RetrievalMode | None = RetrievalMode(args.mode) if args.mode else None

    # ---- Run ID and timestamp ----
    now = datetime.now(timezone.utc)
    run_id = f"eval_{now.strftime('%Y%m%d_%H%M%S')}"
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # ---- Determine effective mode label ----
    mode_label = args.mode or settings.retrieval_mode or "mix"

    # ---- Execute ----
    if args.compare:
        compare_results = run_compare(
            entries,
            settings,
            top_k=top_k,
            generate_answer=generate_answer,
            forced_mode=forced_mode,
        )

        logger.info("Compare run complete. %d entries processed.", len(compare_results))

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(compare_results, f, indent=2, default=str)
        logger.info("Compare results saved → %s", args.output)

        # Print simple summary for compare mode
        print(f"\n{'=' * 60}")
        print("  COMPARE SUMMARY")
        print(f"{'=' * 60}")
        for cr in compare_results:
            g_r = cr["graph"].get("entity_recall") or 0.0
            c_r = cr["chunks_only"].get("entity_recall") or 0.0
            print(f"  • {cr['query'][:55]}... → graph:{g_r:.2f}  chunks:{c_r:.2f}")
        print()

    else:
        # ---- Single workflow run ----
        logger.info("Building graph-enhanced workflow...")
        workflow = build_workflow(settings, enable_graph=True)
        if forced_mode is not None:
            _force_mode_patch(workflow, forced_mode)
            logger.info("Forced retrieval mode: %s", forced_mode.value)

        results: list[dict[str, Any]] = []
        total_to_run = len(entries)

        for i, entry in enumerate(entries, 1):
            logger.info("[%d/%d] Running: %s", i, total_to_run, entry.get("query", "")[:80])
            result = run_single(workflow, entry, top_k=top_k, generate_answer=generate_answer)
            results.append(result)

            # Log per-entry summary
            recall = result.get("entity_recall")
            recall_str = f"{recall:.2f}" if recall is not None else "N/A"
            logger.info(
                "  → elapsed=%.2fs  recall=%s  citations=%d  graph_ents=%d",
                result["elapsed_s"],
                recall_str,
                result["num_citations"],
                result["graph_entities_count"],
            )

        # ---- Summary ----
        summary = compute_summary(
            results,
            mode=mode_label,
            top_k=top_k,
            graph_enabled=True,
            limit=args.limit,
        )

        # ---- Full output ----
        output_payload: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": timestamp,
            "config": {
                "mode": mode_label,
                "top_k": top_k,
                "graph_enabled": True,
                "total_entries": total_entries,
                "limit": args.limit,
                "intent_filter": args.intent,
                "dataset": dataset_path,
            },
            "summary": summary,
            "results": [{k: v for k, v in r.items() if k != "retrieved_contexts"} for r in results],
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, default=str)
        logger.info("Full results saved → %s", args.output)

        # ---- RAGAS output ----
        if args.ragas_output:
            ragas_rows = [to_ragas_row(result, entry) for result, entry in zip(results, entries)]
            with open(args.ragas_output, "w", encoding="utf-8") as f:
                json.dump(ragas_rows, f, indent=2, default=str)
            logger.info("RAGAS output saved → %s", args.ragas_output)

        # ---- Scorecard ----
        verdicts = check_thresholds(summary)

        print_scorecard(
            summary,
            run_id=run_id,
            timestamp=timestamp,
            mode=mode_label,
            top_k=top_k,
            verdicts=verdicts,
        )

        # ---- Markdown report ----
        if args.markdown:
            write_markdown_report(
                summary,
                results,
                run_id=run_id,
                timestamp=timestamp,
                mode=mode_label,
                top_k=top_k,
                verdicts=verdicts,
                output_path=args.markdown,
            )

        # ---- Console summary ----
        print(f"\nOutput:       {args.output}")
        if args.ragas_output:
            print(f"RAGAS output: {args.ragas_output}")
        if args.markdown:
            print(f"Markdown:     {args.markdown}")
        print(f"Run ID:       {run_id}")

        all_pass = all(v["passed"] for v in verdicts)
        if not all_pass:
            print("\n⚠️  Some threshold gates FAILED. Review the scorecard above.")
        print()

    tracing_svc.shutdown()


if __name__ == "__main__":
    main()
