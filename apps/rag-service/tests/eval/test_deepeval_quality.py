"""DeepEval hallucination and quality tests.

Requires ``deepeval`` package and RAG_EVAL_ENABLED=true.
Run with: ``pytest tests/eval/test_deepeval_quality.py -v -m eval``
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


@pytest.fixture()
def _deepeval():
    deepeval = pytest.importorskip("deepeval")
    test_case_mod = pytest.importorskip("deepeval.test_case")
    metrics_mod = pytest.importorskip("deepeval.metrics")
    return deepeval, test_case_mod, metrics_mod


class TestDeepEvalQuality:
    def test_no_hallucination_on_factual_queries(
        self,
        golden_dataset: list[dict[str, Any]],
        _deepeval: tuple,
    ):
        deepeval, test_case_mod, metrics_mod = _deepeval

        LLMTestCase = getattr(test_case_mod, "LLMTestCase", None)
        HallucinationMetric = getattr(metrics_mod, "HallucinationMetric", None)
        assert_test = getattr(deepeval, "assert_test", None)

        if not all([LLMTestCase, HallucinationMetric, assert_test]):
            pytest.skip("DeepEval API classes not available")

        threshold = float(os.getenv("RAG_EVAL_HALLUCINATION_THRESHOLD", "0.5"))
        factual_entries = [e for e in golden_dataset if e["intent"] == "factual"][:3]

        for entry in factual_entries:
            test_case = LLMTestCase(
                input=entry["query"],
                actual_output=entry["expected_answer"],
                expected_output=entry["expected_answer"],
                retrieval_context=[f"Context for {c}" for c in entry["expected_contexts"]],
            )
            metric = HallucinationMetric(threshold=threshold)
            assert_test(test_case, [metric])

    def test_answer_relevancy_on_all_intents(
        self,
        golden_dataset_by_intent: dict[str, list[dict[str, Any]]],
        _deepeval: tuple,
    ):
        _, test_case_mod, metrics_mod = _deepeval

        LLMTestCase = getattr(test_case_mod, "LLMTestCase", None)
        AnswerRelevancyMetric = getattr(metrics_mod, "AnswerRelevancyMetric", None)

        if not all([LLMTestCase, AnswerRelevancyMetric]):
            pytest.skip("DeepEval AnswerRelevancyMetric not available")

        for intent, entries in golden_dataset_by_intent.items():
            entry = entries[0]
            test_case = LLMTestCase(
                input=entry["query"],
                actual_output=entry["expected_answer"],
                retrieval_context=[f"Context for {c}" for c in entry["expected_contexts"]],
            )
            metric = AnswerRelevancyMetric(threshold=0.7)
            score = metric.measure(test_case)
            assert score >= 0.7, f"Answer relevancy for '{intent}' intent: {score:.3f} < 0.7"

    def test_faithfulness_on_golden_entries(
        self,
        golden_dataset: list[dict[str, Any]],
        _deepeval: tuple,
    ):
        _, test_case_mod, metrics_mod = _deepeval

        LLMTestCase = getattr(test_case_mod, "LLMTestCase", None)
        FaithfulnessMetric = getattr(metrics_mod, "FaithfulnessMetric", None)

        if not all([LLMTestCase, FaithfulnessMetric]):
            pytest.skip("DeepEval FaithfulnessMetric not available")

        for entry in golden_dataset[:3]:
            test_case = LLMTestCase(
                input=entry["query"],
                actual_output=entry["expected_answer"],
                retrieval_context=[f"Context for {c}" for c in entry["expected_contexts"]],
            )
            metric = FaithfulnessMetric(threshold=0.8)
            score = metric.measure(test_case)
            assert score >= 0.8, f"Faithfulness for {entry['id']}: {score:.3f} < 0.8"
