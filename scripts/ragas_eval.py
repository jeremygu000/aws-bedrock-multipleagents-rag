#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import fmean
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("JSON input must be an array of objects.")
        return [coerce_row(row) for row in parsed]

    return [coerce_row(json.loads(line)) for line in raw.splitlines() if line.strip()]


_RAGAS_FIELD_KEYS = ("reference", "retrieved_contexts", "ground_truth_context")


def coerce_row(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Each input row must be a JSON object.")
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        for key in _RAGAS_FIELD_KEYS:
            if key not in value and key in metadata:
                value[key] = metadata[key]
    return value


def has_value(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return len(value) > 0
    return True


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_group_key(row: dict[str, Any], group_by: str) -> str:
    if group_by == "none":
        return "all"

    category = row.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        nested_category = metadata.get("category")
        if isinstance(nested_category, str) and nested_category.strip():
            return nested_category.strip()

    return "uncategorized"


def group_rows(rows: list[dict[str, Any]], group_by: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = get_group_key(row, group_by)
        grouped.setdefault(key, []).append(row)
    return grouped


def normalize_group_name(group_name: str) -> str:
    return group_name.strip().lower().replace("_", "-")


def get_default_metric_preferences(group_name: str) -> list[str] | None:
    normalized = normalize_group_name(group_name)

    if normalized == "qa":
        return ["semantic_similarity", "factual_correctness"]

    if normalized in {"work-search", "work"}:
        return ["semantic_similarity"]

    return None


def resolve_region(explicit_region: str | None, provider: str = "bedrock") -> str:
    if provider == "ollama":
        return os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2")

    region = (
        explicit_region
        or os.environ.get("RAGAS_AWS_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if not region:
        raise ValueError("Missing AWS region. Pass --region or set AWS_DEFAULT_REGION.")

    os.environ.setdefault("AWS_REGION", region)
    os.environ.setdefault("AWS_DEFAULT_REGION", region)
    os.environ.setdefault("AWS_REGION_NAME", region)
    return region


def build_evaluator_models(
    llm_model: str,
    embedding_model: str,
    region: str,
    provider: str = "bedrock",
    timeout: int = 300,
) -> tuple[Any, Any, str]:
    if provider == "ollama":
        return _build_ollama_models(llm_model, embedding_model)
    return _build_bedrock_models(llm_model, embedding_model, region, timeout=timeout)


def _patch_embed_query(embeddings: Any) -> None:
    """RAGAS 0.4.3 _answer_relevance calls embed_query but modern providers only have embed_text."""
    if not hasattr(embeddings, "embed_query") and hasattr(embeddings, "embed_text"):
        embeddings.embed_query = embeddings.embed_text
    if not hasattr(embeddings, "aembed_query") and hasattr(embeddings, "aembed_text"):
        embeddings.aembed_query = embeddings.aembed_text


def _build_ollama_models(llm_model: str, embedding_model: str) -> tuple[Any, Any, str]:
    from openai import OpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_id = embedding_model if embedding_model.startswith("ollama") else f"ollama/{embedding_model}"

    client = OpenAI(base_url=f"{ollama_base}/v1", api_key="ollama")
    llm = llm_factory(llm_model, client=client, temperature=0.3)
    embeddings = embedding_factory(
        "litellm", model=embed_id, api_base=ollama_base, timeout=300
    )
    _patch_embed_query(embeddings)
    return llm, embeddings, "ollama"


def _build_bedrock_models(
    llm_model: str, embedding_model: str, region: str, timeout: int = 300
) -> tuple[Any, Any, str]:
    import instructor
    import litellm
    from ragas.embeddings.base import embedding_factory
    from ragas.llms.litellm_llm import LiteLLMStructuredLLM

    # LiteLLM uses "bedrock/<model-id>" format and reads AWS credentials from
    # environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME).
    os.environ.setdefault("AWS_REGION_NAME", region)
    litellm.drop_params = True

    litellm_model = f"bedrock/{llm_model}"
    litellm_embed = f"bedrock/{embedding_model}"

    # FactualCorrectness and other claim-based metrics call `await llm.agenerate()`
    # which requires is_async=True. We create an AsyncInstructor client via
    # instructor.from_litellm(litellm.acompletion) — the class name "AsyncInstructor"
    # is detected by LiteLLMStructuredLLM._check_client_async() to set is_async=True.
    async_client = instructor.from_litellm(litellm.acompletion)
    llm = LiteLLMStructuredLLM(
        client=async_client,
        model=litellm_model,
        provider="litellm",
        temperature=0.1,
        max_tokens=8192,
    )

    embeddings = embedding_factory(
        "litellm", model=litellm_embed, timeout=timeout
    )
    _patch_embed_query(embeddings)
    return llm, embeddings, "litellm-bedrock"


def import_metrics() -> dict[str, Any]:
    try:
        from ragas.metrics.collections import (
            ContextRecall,
            Faithfulness,
            FactualCorrectness,
            ResponseRelevancy,
            SemanticSimilarity,
        )
    except ImportError:
        try:
            from ragas.metrics.collections import (  # type: ignore[no-redef]
                Faithfulness,
                FactualCorrectness,
                LLMContextRecall as ContextRecall,  # type: ignore[no-redef]
                ResponseRelevancy,
                SemanticSimilarity,
            )
        except ImportError:
            from ragas.metrics import (  # type: ignore[no-redef]
                Faithfulness,
                FactualCorrectness,
                LLMContextRecall as ContextRecall,  # type: ignore[no-redef]
                ResponseRelevancy,
                SemanticSimilarity,
            )

    return {
        "response_relevancy": ResponseRelevancy,
        "faithfulness": Faithfulness,
        "context_recall": ContextRecall,
        "factual_correctness": FactualCorrectness,
        "semantic_similarity": SemanticSimilarity,
    }


def pick_metric_names(
    rows: list[dict[str, Any]],
    requested_metrics: list[str] | None,
    group_name: str,
) -> tuple[list[str], list[str]]:
    requirements = {
        "response_relevancy": ("user_input", "response"),
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "context_recall": ("user_input", "reference", "retrieved_contexts"),
        "factual_correctness": ("user_input", "response", "reference"),
        "semantic_similarity": ("response", "reference"),
    }

    available = [
        metric_name
        for metric_name, keys in requirements.items()
        if all(all(has_value(row, key) for key in keys) for row in rows)
    ]

    if requested_metrics:
        unsupported = [metric for metric in requested_metrics if metric not in requirements]
        if unsupported:
            raise ValueError(
                "Unsupported metrics: "
                + ", ".join(sorted(unsupported))
                + ". Supported metrics: "
                + ", ".join(sorted(requirements))
            )

        missing_inputs = [
            metric_name
            for metric_name in requested_metrics
            if metric_name not in available
        ]
        if missing_inputs:
            raise ValueError(
                "Requested metrics are missing required fields in at least one row: "
                + ", ".join(missing_inputs)
            )

        return requested_metrics, available

    preferred_metrics = get_default_metric_preferences(group_name)
    if preferred_metrics:
        selected = [metric_name for metric_name in preferred_metrics if metric_name in available]
        if selected:
            return selected, available

    return available, available


def build_metrics(metric_names: list[str], llm: Any, embeddings: Any) -> list[Any]:
    metric_classes = import_metrics()
    metrics: list[Any] = []

    for metric_name in metric_names:
        metric_class = metric_classes[metric_name]
        if metric_name == "response_relevancy":
            metrics.append(metric_class(llm=llm, embeddings=embeddings))
        elif metric_name == "semantic_similarity":
            metrics.append(metric_class(embeddings=embeddings))
        elif metric_name in {"faithfulness", "context_recall", "factual_correctness"}:
            metrics.append(metric_class(llm=llm))
        else:
            raise ValueError(f"Unsupported metric configuration: {metric_name}")

    return metrics


def _resolve_metric_column(rows: list[dict[str, Any]], metric_name: str) -> str | None:
    """RAGAS may suffix column names with mode info, e.g. 'factual_correctness(mode=f1)'.
    Find the actual column key that starts with metric_name.
    """
    if not rows:
        return metric_name
    first = rows[0]
    if metric_name in first:
        return metric_name
    for key in first:
        if key.startswith(metric_name):
            return key
    return None


def score_metric(
    rows: list[dict[str, Any]], metric: Any, timeout: int = 300
) -> tuple[list[dict[str, Any]], float | None]:
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=timeout, max_retries=3, max_workers=8, max_wait=60)
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset=dataset,
        metrics=[metric],
        run_config=run_config,
        raise_exceptions=False,
    )
    metric_name = str(getattr(metric, "name", metric.__class__.__name__))
    scored_rows = [normalize_json_value(row) for row in result.to_pandas().to_dict(orient="records")]

    resolved_key = _resolve_metric_column(scored_rows, metric_name)
    if resolved_key and resolved_key != metric_name:
        for row in scored_rows:
            if resolved_key in row:
                row[metric_name] = row.pop(resolved_key)

    values = [
        float(row[metric_name])
        for row in scored_rows
        if isinstance(row.get(metric_name), (int, float))
        and not math.isnan(float(row[metric_name]))
    ]
    return scored_rows, (fmean(values) if values else None)


def score_rows(
    rows: list[dict[str, Any]], metrics: list[Any], timeout: int = 300
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, str]]:
    scored_rows = [normalize_json_value(row) for row in rows]
    summary: dict[str, float] = {}
    failures: dict[str, str] = {}

    for metric in metrics:
        metric_name = str(getattr(metric, "name", metric.__class__.__name__))

        try:
            metric_rows, average = score_metric(rows, metric, timeout=timeout)
        except Exception as error:
            failures[metric_name] = str(error)
            for row in scored_rows:
                row[metric_name] = None
            continue

        for index, metric_row in enumerate(metric_rows):
            scored_rows[index][metric_name] = metric_row.get(metric_name)

        if average is not None:
            summary[metric_name] = average

    return scored_rows, summary, failures


def parse_metric_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    metrics = [item.strip() for item in raw.split(",") if item.strip()]
    return metrics or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAGAS on a ragas-shaped JSON or JSONL dataset, typically generated by "
            "`pnpm eval:agent -- --output-format ragas`."
        )
    )
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL input dataset.")
    parser.add_argument("--output", required=True, help="Path to the JSON results file.")
    parser.add_argument(
        "--group-by",
        default="category",
        help="Grouping strategy. Use category or none. Defaults to category.",
    )
    parser.add_argument(
        "--region",
        help="AWS region for evaluator models. Defaults to RAGAS_AWS_REGION, AWS_REGION, or AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("RAGAS_EVAL_LLM_MODEL"),
        help="Bedrock evaluator LLM model id. Can also come from RAGAS_EVAL_LLM_MODEL.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("RAGAS_EVAL_EMBEDDING_MODEL"),
        help="Bedrock embedding model id. Can also come from RAGAS_EVAL_EMBEDDING_MODEL.",
    )
    parser.add_argument(
        "--metrics",
        help=(
            "Comma-separated metrics. Defaults to auto-select from available fields. "
            "Supported: response_relevancy, faithfulness, context_recall, factual_correctness, semantic_similarity."
        ),
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("RAGAS_EVAL_PROVIDER", "bedrock"),
        choices=["bedrock", "ollama"],
        help="Evaluator provider: bedrock (default) or ollama for local models.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("RAGAS_EVAL_TIMEOUT", "300")),
        help="Timeout in seconds for each RAGAS metric evaluation job and Bedrock API calls. Defaults to 300.",
    )
    raw_args = sys.argv[1:]
    args = parser.parse_args(raw_args[1:] if raw_args and raw_args[0] == "--" else raw_args)

    if args.provider == "ollama":
        if not args.llm_model or args.llm_model.startswith(("amazon.", "anthropic.", "bedrock/")):
            args.llm_model = "qwen3:32b"
        if not args.embedding_model or args.embedding_model.startswith(("amazon.", "bedrock/")):
            args.embedding_model = "nomic-embed-text:latest"
    else:
        if not args.llm_model:
            raise ValueError(
                "Missing evaluator LLM model. Pass --llm-model or set RAGAS_EVAL_LLM_MODEL."
            )
        if not args.embedding_model:
            raise ValueError(
                "Missing evaluator embedding model. Pass --embedding-model or set RAGAS_EVAL_EMBEDDING_MODEL."
            )
    if args.group_by not in {"category", "none"}:
        raise ValueError("--group-by must be either category or none.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(input_path)
    if not rows:
        raise ValueError("Input dataset is empty.")

    region = resolve_region(args.region, args.provider)
    llm, embeddings, provider = build_evaluator_models(
        args.llm_model, args.embedding_model, region, args.provider, timeout=args.timeout
    )
    requested_metrics = parse_metric_list(args.metrics)
    grouped = group_rows(rows, args.group_by)
    groups_output: dict[str, Any] = {}
    flattened_rows: list[dict[str, Any]] = []
    summary_by_group: dict[str, dict[str, float]] = {}
    failures_by_group: dict[str, dict[str, str]] = {}

    for group_name, group_rows_list in grouped.items():
        metric_names, available_metrics = pick_metric_names(
            group_rows_list, requested_metrics, group_name
        )
        if not metric_names:
            groups_output[group_name] = {
                "row_count": len(group_rows_list),
                "selected_metrics": [],
                "auto_available_metrics": available_metrics,
                "summary": {},
                "metric_failures": {
                    "_group": (
                        "No compatible metrics could be selected from this group. "
                        "Expected fields such as user_input, response, reference, and retrieved_contexts."
                    )
                },
                "rows": [normalize_json_value(row) for row in group_rows_list],
            }
            failures_by_group[group_name] = groups_output[group_name]["metric_failures"]
            for row in groups_output[group_name]["rows"]:
                row["evaluation_group"] = group_name
                flattened_rows.append(row)
            continue

        metrics = build_metrics(metric_names, llm, embeddings)
        scored_rows, summary, failures = score_rows(group_rows_list, metrics, timeout=args.timeout)
        for row in scored_rows:
            row["evaluation_group"] = group_name
            flattened_rows.append(row)

        groups_output[group_name] = {
            "row_count": len(group_rows_list),
            "selected_metrics": metric_names,
            "auto_available_metrics": available_metrics,
            "summary": summary,
            "metric_failures": failures,
            "rows": scored_rows,
        }
        summary_by_group[group_name] = summary
        failures_by_group[group_name] = failures

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "row_count": len(rows),
                "provider": provider,
                "region": region,
                "llm_model": args.llm_model,
                "embedding_model": args.embedding_model,
                "group_by": args.group_by,
                "group_count": len(grouped),
                "summary_by_group": summary_by_group,
                "metric_failures_by_group": failures_by_group,
                "groups": groups_output,
                "rows": flattened_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved: {output_path.resolve()}")
    print(f"rows: {len(rows)}")
    print(f"provider: {provider}")
    print(f"group by: {args.group_by}")
    for group_name, group_output in groups_output.items():
        print(f"[group:{group_name}] rows={group_output['row_count']}")
        selected_metrics = group_output["selected_metrics"]
        print(
            f"[group:{group_name}] selected metrics: "
            f"{', '.join(selected_metrics) if selected_metrics else 'none'}"
        )
        for metric_name, value in group_output["summary"].items():
            print(f"[group:{group_name}] {metric_name}: {value:.4f}")
        for metric_name, message in group_output["metric_failures"].items():
            print(f"[group:{group_name}] metric failed: {metric_name}: {message}")


if __name__ == "__main__":
    main()
