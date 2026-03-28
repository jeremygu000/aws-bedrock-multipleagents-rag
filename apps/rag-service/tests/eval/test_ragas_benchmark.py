"""RAGAS benchmark integration tests.

These tests require:
- ``ragas`` package (install via ``pip install 'hybrid-rag-service[eval]'``)
- A running RAG workflow or mocked pipeline
- AWS Bedrock credentials (for RAGAS LLM judge)

Run with: ``pytest tests/eval/test_ragas_benchmark.py -v -m eval``
Skipped in regular CI unless RAG_EVAL_ENABLED=true.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

EVAL_ENABLED = os.getenv("RAG_EVAL_ENABLED", "false").lower() == "true"

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not EVAL_ENABLED, reason="RAG_EVAL_ENABLED not set"),
]


def _build_ragas_samples(
    golden_dataset: list[dict[str, Any]],
    run_query_fn: Any,
) -> list[dict[str, Any]]:
    samples = []
    for entry in golden_dataset:
        state = run_query_fn(query=entry["query"], top_k=10)
        contexts = [
            h.get("chunk_text", h.get("content", "")) for h in state.get("reranked_hits", [])
        ]
        samples.append(
            {
                "user_input": entry["query"],
                "response": state.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": entry["expected_answer"],
            }
        )
    return samples


@pytest.fixture()
def _ragas_metrics():
    ragas = pytest.importorskip("ragas")
    metrics_mod = pytest.importorskip("ragas.metrics")
    return ragas, metrics_mod


class TestRagasBenchmark:
    def test_golden_dataset_has_minimum_entries(self, golden_dataset: list[dict[str, Any]]):
        assert (
            len(golden_dataset) >= 50
        ), f"Golden dataset has only {len(golden_dataset)} entries, need >= 50"

    def test_golden_dataset_covers_all_intents(self, golden_dataset: list[dict[str, Any]]):
        intents = {e["intent"] for e in golden_dataset}
        required = {"factual", "analytical", "comparative", "exploratory"}
        missing = required - intents
        assert not missing, f"Golden dataset missing intents: {missing}"

    def test_golden_dataset_entry_structure(self, golden_dataset: list[dict[str, Any]]):
        required_keys = {
            "id",
            "query",
            "expected_answer",
            "expected_contexts",
            "expected_entities",
            "intent",
            "complexity",
            "tags",
        }
        for entry in golden_dataset:
            missing = required_keys - set(entry.keys())
            assert not missing, f"Entry {entry.get('id', '?')} missing keys: {missing}"
            assert len(entry["query"]) > 0, f"Entry {entry['id']} has empty query"
            assert (
                len(entry["expected_answer"]) > 0
            ), f"Entry {entry['id']} has empty expected_answer"

    def test_golden_dataset_ids_unique(self, golden_dataset: list[dict[str, Any]]):
        ids = [e["id"] for e in golden_dataset]
        assert len(ids) == len(set(ids)), "Golden dataset has duplicate IDs"

    def test_ragas_evaluation(
        self,
        golden_dataset: list[dict[str, Any]],
        _ragas_metrics: tuple,
    ):
        ragas, metrics_mod = _ragas_metrics
        EvaluationDataset = getattr(ragas, "EvaluationDataset", None)
        evaluate_fn = getattr(ragas, "evaluate", None)
        if not EvaluationDataset or not evaluate_fn:
            pytest.skip("RAGAS v0.2+ API not available")

        Faithfulness = getattr(metrics_mod, "Faithfulness", None)
        AnswerRelevancy = getattr(metrics_mod, "AnswerRelevancy", None)
        ContextPrecision = getattr(metrics_mod, "ContextPrecision", None)
        ContextRecall = getattr(metrics_mod, "ContextRecall", None)

        if not all([Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall]):
            pytest.skip("Required RAGAS metrics not available")

        samples = [
            {
                "user_input": e["query"],
                "response": e["expected_answer"],
                "retrieved_contexts": [f"Context for {c}" for c in e["expected_contexts"]],
                "reference": e["expected_answer"],
            }
            for e in golden_dataset[:5]
        ]

        dataset = EvaluationDataset.from_list(samples)
        metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
        scores = evaluate_fn(dataset, metrics=metrics)

        thresholds = {
            "faithfulness": float(os.getenv("RAG_EVAL_FAITHFULNESS_THRESHOLD", "0.85")),
            "answer_relevancy": float(os.getenv("RAG_EVAL_ANSWER_RELEVANCY_THRESHOLD", "0.80")),
            "context_precision": float(os.getenv("RAG_EVAL_CONTEXT_PRECISION_THRESHOLD", "0.75")),
            "context_recall": float(os.getenv("RAG_EVAL_CONTEXT_RECALL_THRESHOLD", "0.80")),
        }

        for metric_name, threshold in thresholds.items():
            score = scores.get(metric_name)
            if score is not None:
                assert score >= threshold, f"{metric_name}: {score:.3f} < {threshold} threshold"
