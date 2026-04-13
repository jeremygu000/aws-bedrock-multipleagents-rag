#!/usr/bin/env python3
"""
RAGAS baseline evaluation runner for feature benchmarking.
Supports 6 feature configurations: all-disabled → each feature individually → all-enabled.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add apps/rag-service to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "rag-service"))

from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    SemanticSimilarity,
)
from datasets import Dataset
from botocore.config import Config
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_REGION = "ap-southeast-2"
DEFAULT_LLM_MODEL = "amazon.nova-pro-v1:0"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_TIMEOUT_S = 300


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    """Load dataset from JSONL file."""
    rows = []
    with open(file_path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def get_metrics_for_category(category: str):
    """Get appropriate metrics for category (QA vs work-search)."""
    if category == "qa":
        return [SemanticSimilarity(), FactualCorrectness()]
    else:  # work-search and other categories
        return [SemanticSimilarity()]


def evaluation_result_to_dict(result):
    """Convert RAGAS EvaluationResult to JSON-serializable dict."""
    output = {}
    
    # Top-level scores by metric
    if hasattr(result, 'scores'):
        for key, value in result.scores.items():
            if isinstance(value, float):
                output[f"overall_{key}"] = round(value, 4)
    
    # Per-row detailed scores
    if hasattr(result, 'dataset'):
        output['rows_count'] = len(result.dataset)
    
    # Add per-metric scores if available
    for metric_name in ['semantic_similarity', 'factual_correctness']:
        if hasattr(result, metric_name):
            val = getattr(result, metric_name)
            if isinstance(val, (int, float)):
                output[metric_name] = round(val, 4)
    
    return output if output else {"status": "evaluation_completed"}


def run_ragas_evaluation(
    dataset: Dataset,
    category: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict:
    """Run RAGAS evaluation for a category."""
    print(f"\n  → Running RAGAS evaluation for {category}...")
    print(f"    Rows: {len(dataset)}")
    print(f"    Judge LLM: {llm_model}")
    print(f"    Embedding: {embedding_model}")
    print(f"    Timeout: {timeout_s}s")

    # Create botocore config with timeout
    bedrock_config = Config(
        connect_timeout=timeout_s,
        read_timeout=timeout_s,
        retries={"max_attempts": 3},
    )

    # Initialize Bedrock LLM
    llm = ChatBedrock(
        model_id=llm_model,
        region_name=DEFAULT_REGION,
        config=bedrock_config,
    )

    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(
        model_id=embedding_model,
        region_name=DEFAULT_REGION,
        config=bedrock_config,
    )

    # Wrap for RAGAS
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Get metrics for this category
    metrics = get_metrics_for_category(category)

    # Run evaluation
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        # Convert to JSON-serializable format
        return evaluation_result_to_dict(results)
    except Exception as e:
        print(f"    ❌ Error during evaluation: {e}")
        raise


def main():
    """Main evaluation runner."""
    print("=" * 80)
    print("RAGAS BASELINE EVALUATION")
    print("=" * 80)

    # Parse arguments
    input_file = os.getenv("RAGAS_INPUT_FILE", "scripts/examples/agent-eval.example.jsonl")
    output_dir = os.getenv("RAGAS_OUTPUT_DIR", "benchmarks/ragas-baselines")
    feature_config = os.getenv("RAGAS_FEATURE_CONFIG", "all-disabled")
    llm_model = os.getenv("RAGAS_EVAL_LLM_MODEL", DEFAULT_LLM_MODEL)
    embedding_model = os.getenv("RAGAS_EVAL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    timeout_s = int(os.getenv("RAGAS_EVAL_TIMEOUT_S", DEFAULT_TIMEOUT_S))

    # Load dataset
    print(f"\nLoading dataset from {input_file}...")
    dataset = load_dataset_from_jsonl(input_file)
    print(f"  → Total rows: {len(dataset)}")

    # Group by category
    qa_rows = [r for r in dataset if r.get("metadata", {}).get("category") == "qa"]
    work_search_rows = [r for r in dataset if r.get("metadata", {}).get("category") == "work-search"]

    print(f"  → QA rows: {len(qa_rows)}")
    print(f"  → Work-search rows: {len(work_search_rows)}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run evaluations per category
    results_by_category = {}

    if qa_rows:
        print("\n" + "─" * 80)
        print("📊 QA EVALUATION")
        print("─" * 80)
        qa_dataset = Dataset.from_list(qa_rows)
        qa_results = run_ragas_evaluation(
            qa_dataset, "qa", llm_model, embedding_model, timeout_s
        )
        results_by_category["qa"] = qa_results
        print(f"    ✅ Results: {qa_results}")

    if work_search_rows:
        print("\n" + "─" * 80)
        print("📊 WORK-SEARCH EVALUATION")
        print("─" * 80)
        ws_dataset = Dataset.from_list(work_search_rows)
        ws_results = run_ragas_evaluation(
            ws_dataset, "work-search", llm_model, embedding_model, timeout_s
        )
        results_by_category["work-search"] = ws_results
        print(f"    ✅ Results: {ws_results}")

    # Write output
    output_file = f"{output_dir}/baseline-{feature_config}-2026-04-12.json"
    output_data = {
        "feature_configuration": feature_config,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "results_by_category": results_by_category,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results written to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
