from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

EVAL_ENABLED = os.getenv("RAG_EVAL_ENABLED", "false").lower() == "true"


@pytest.fixture(scope="session")
def golden_dataset() -> list[dict[str, Any]]:
    with open(GOLDEN_DATASET_PATH) as f:
        data = json.load(f)
    return data["entries"]


@pytest.fixture(scope="session")
def golden_dataset_by_intent(
    golden_dataset: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for entry in golden_dataset:
        intent = entry["intent"]
        by_intent.setdefault(intent, []).append(entry)
    return by_intent


def _make_workflow_state(
    *,
    query: str = "test query",
    answer: str = "test answer",
    reranked_hits: list[dict[str, Any]] | None = None,
    graph_context: Any = None,
    cache_hit: bool = False,
) -> dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "reranked_hits": reranked_hits or [],
        "graph_context": graph_context,
        "cache_hit": cache_hit,
    }


@pytest.fixture(scope="session")
def pipeline_results(
    golden_dataset: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run all golden entries through the real RagWorkflow pipeline.

    Returns a dict mapping entry["id"] -> result dict (from run_single).
    Gated by RAG_EVAL_ENABLED=true; returns empty dict otherwise so the
    tests themselves can skip via their own skipif markers.
    """
    if not EVAL_ENABLED:
        return {}

    # Lazy imports — only inside the fixture to avoid import errors when
    # eval deps are not installed or infra is not available.
    from app.config import get_settings  # noqa: PLC0415
    from app.tracing import init_tracing  # noqa: PLC0415
    from scripts.eval_runner import run_single  # noqa: PLC0415
    from scripts.test_rag_queries import build_workflow  # noqa: PLC0415

    settings = get_settings()
    init_tracing(settings)
    workflow = build_workflow(settings, enable_graph=True)

    results: dict[str, dict[str, Any]] = {}
    for entry in golden_dataset:
        result = run_single(workflow, entry)
        results[entry["id"]] = result

    return results


def _pipeline_entry_result(
    pipeline_results: dict[str, dict[str, Any]],
    entry_id: str,
) -> dict[str, Any] | None:
    """Return the pipeline result for a specific entry ID, or None if missing."""
    return pipeline_results.get(entry_id)
