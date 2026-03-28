from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


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
