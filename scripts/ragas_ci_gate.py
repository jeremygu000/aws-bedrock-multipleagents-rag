#!/usr/bin/env python3
"""RAGAS CI/CD quality gate — compare evaluation results against baselines and thresholds.

Loads RAGAS results (from ragas_eval.py), baseline metrics, and threshold config.
Outputs pass/fail report. Exit code 0 = pass, 1 = fail, 2 = configuration error.

Usage:
    python3 scripts/ragas_ci_gate.py \\
      --results tmp/evals/supervisor-ragas-results.json \\
      --thresholds eval/thresholds.json \\
      --baseline eval/baselines/metrics-v1.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# ─── Data types ────────────────────────────────────────────────────────────────

CheckResult = dict[str, Any]  # {group, metric, value, threshold?, baseline?, reason, passed, improved}


# ─── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {label} not found: {path}", file=sys.stderr)
        sys.exit(2)
    except OSError as exc:
        print(f"ERROR: Cannot read {label} ({path}): {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Malformed JSON in {label} ({path}): {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(parsed, dict):
        print(f"ERROR: {label} must be a JSON object, got {type(parsed).__name__}", file=sys.stderr)
        sys.exit(2)

    return parsed  # type: ignore[return-value]


def try_load_json(path: Path, label: str) -> dict[str, Any] | None:
    """Load JSON, returning None (with a warning) if the file is missing."""
    if not path.exists():
        print(f"WARNING: {label} not found ({path}) — regression checks skipped.")
        return None
    return load_json(path, label)


# ─── Threshold lookup ──────────────────────────────────────────────────────────

def get_threshold(
    thresholds: dict[str, Any],
    group: str,
    metric: str,
) -> float | None:
    """Return group-specific threshold, falling back to defaults."""
    group_thresholds: dict[str, Any] = thresholds.get("groups", {}).get(group, {})
    if metric in group_thresholds:
        return float(group_thresholds[metric])
    defaults: dict[str, Any] = thresholds.get("defaults", {})
    if metric in defaults:
        return float(defaults[metric])
    return None


# ─── Gate logic ────────────────────────────────────────────────────────────────

def check_metric(
    group: str,
    metric: str,
    value: float,
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None,
    strict: bool,
) -> CheckResult:
    result: CheckResult = {
        "group": group,
        "metric": metric,
        "value": value,
        "passed": True,
        "improved": False,
        "reason": None,
        "threshold": None,
        "baseline_mean": None,
        "baseline_tolerance": None,
    }

    # 1. NaN check
    if math.isnan(value):
        result["passed"] = False if strict else result["passed"]
        result["reason"] = "nan_value"
        if strict:
            result["passed"] = False
        else:
            # warn only — leave passed as True but flag reason
            result["reason"] = "nan_value_warning"
        return result

    # 2. Absolute threshold check
    threshold = get_threshold(thresholds, group, metric)
    result["threshold"] = threshold
    if threshold is not None and value < threshold:
        result["passed"] = False
        result["reason"] = "below_threshold"

    # 3. Baseline regression check
    if baseline is not None:
        baseline_groups: dict[str, Any] = baseline.get("groups", {})
        group_baseline = baseline_groups.get(group, {})
        if metric in group_baseline:
            entry = group_baseline[metric]
            mean = float(entry.get("mean", 0))
            tolerance = float(entry.get("tolerance", 0))
            result["baseline_mean"] = mean
            result["baseline_tolerance"] = tolerance

            lower = mean - tolerance
            upper = mean + tolerance

            if value < lower:
                result["passed"] = False
                # Only override reason if we haven't already flagged threshold failure
                if result["reason"] is None:
                    result["reason"] = "baseline_regression"

            elif value > upper:
                result["improved"] = True

    # Set reason for clean passes
    if result["passed"] and result["reason"] is None:
        result["reason"] = "passed"

    return result


def run_gate(
    results: dict[str, Any],
    thresholds: dict[str, Any],
    baseline: dict[str, Any] | None,
    strict: bool,
) -> list[CheckResult]:
    summary_by_group: dict[str, Any] = results.get("summary_by_group", {})
    groups_meta: dict[str, Any] = results.get("groups", {})

    checks: list[CheckResult] = []

    for group, metrics in summary_by_group.items():
        if not isinstance(metrics, dict):
            continue
        for metric, raw_value in metrics.items():
            try:
                value = float(raw_value) if raw_value is not None else float("nan")
            except (TypeError, ValueError):
                value = float("nan")

            check = check_metric(group, metric, value, thresholds, baseline, strict)
            # Attach row count from groups metadata
            group_info = groups_meta.get(group, {})
            check["row_count"] = group_info.get("row_count", group_info.get("count", "?"))
            checks.append(check)

    return checks


# ─── Report formatting ─────────────────────────────────────────────────────────

def format_metric_line(check: CheckResult) -> str:
    value = check["value"]
    metric = check["metric"]
    threshold = check["threshold"]
    mean = check["baseline_mean"]
    tolerance = check["baseline_tolerance"]
    passed = check["passed"]
    improved = check["improved"]
    reason = check["reason"]

    if reason in ("nan_value", "nan_value_warning"):
        icon = "⚠️ " if reason == "nan_value_warning" else "❌"
        note = " — NaN detected (warn)" if reason == "nan_value_warning" else " — NaN FAILED"
        return f"  {icon} {metric}: NaN{note}"

    icon = "✅" if passed else "❌"
    val_str = f"{value:.4f}".rstrip("0").rstrip(".")

    details: list[str] = []
    if threshold is not None:
        details.append(f"threshold: {threshold}")
    if mean is not None and tolerance is not None:
        details.append(f"baseline: {mean} ± {tolerance}")

    detail_str = f"  ({', '.join(details)})" if details else ""
    suffix = ""
    if improved:
        suffix = " — improved!"
    elif not passed and reason == "below_threshold":
        suffix = " — BELOW THRESHOLD"
    elif not passed and reason == "baseline_regression":
        suffix = " — REGRESSION vs baseline"
    elif not passed and reason and "nan" not in reason:
        suffix = f" — {reason.upper()}"

    return f"  {icon} {metric}: {val_str}{detail_str}{suffix}"


def print_report(checks: list[CheckResult]) -> None:
    print("═" * 43)
    print("  RAGAS CI/CD Quality Gate Report")
    print("═" * 43)
    print()

    # Group checks by group name
    groups_seen: dict[str, list[CheckResult]] = {}
    for check in checks:
        g = check["group"]
        groups_seen.setdefault(g, []).append(check)

    for group, group_checks in groups_seen.items():
        row_count = group_checks[0].get("row_count", "?") if group_checks else "?"
        print(f"Group: {group} ({row_count} rows)")
        for check in group_checks:
            print(format_metric_line(check))
        print()

    failed = [c for c in checks if not c["passed"]]
    total = len(checks)
    n_failed = len(failed)
    n_passed = total - n_failed

    print("─" * 43)
    if n_failed == 0:
        print(f"  RESULT: ✅ PASSED ({n_passed}/{total} checks passed)")
    else:
        plural = "check" if n_failed == 1 else "checks"
        print(f"  RESULT: ❌ FAILED ({n_failed} {plural} failed)")
    print("─" * 43)


def build_json_report(checks: list[CheckResult]) -> dict[str, Any]:
    failures = [
        {
            "group": c["group"],
            "metric": c["metric"],
            "value": c["value"],
            "threshold": c["threshold"],
            "baseline_mean": c["baseline_mean"],
            "baseline_tolerance": c["baseline_tolerance"],
            "reason": c["reason"],
        }
        for c in checks
        if not c["passed"]
    ]
    improvements = [
        {
            "group": c["group"],
            "metric": c["metric"],
            "value": c["value"],
            "baseline_mean": c["baseline_mean"],
            "baseline_tolerance": c["baseline_tolerance"],
        }
        for c in checks
        if c["improved"]
    ]

    groups_summary: dict[str, Any] = {}
    for check in checks:
        g = check["group"]
        groups_summary.setdefault(g, {})[check["metric"]] = {
            "value": check["value"],
            "passed": check["passed"],
            "improved": check["improved"],
            "reason": check["reason"],
            "threshold": check["threshold"],
            "baseline_mean": check["baseline_mean"],
            "baseline_tolerance": check["baseline_tolerance"],
        }

    n_failed = sum(1 for c in checks if not c["passed"])
    total = len(checks)

    return {
        "passed": n_failed == 0,
        "checks_total": total,
        "checks_passed": total - n_failed,
        "checks_failed": n_failed,
        "failures": failures,
        "improvements": improvements,
        "groups": groups_summary,
    }


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAGAS CI/CD quality gate — compare evaluation results against baselines and thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        metavar="PATH",
        help="Path to RAGAS results JSON (output from ragas_eval.py)",
    )
    parser.add_argument(
        "--thresholds",
        required=True,
        metavar="PATH",
        help="Path to thresholds config JSON",
    )
    parser.add_argument(
        "--baseline",
        metavar="PATH",
        default=None,
        help="Path to baseline metrics JSON. If missing, regression checks are skipped.",
    )
    parser.add_argument(
        "--json-output",
        metavar="PATH",
        default=None,
        help="Write machine-readable JSON report to this file (optional)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail on any NaN metric (default: warn only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_path = Path(args.results)
    thresholds_path = Path(args.thresholds)

    results = load_json(results_path, "RAGAS results")
    thresholds = load_json(thresholds_path, "thresholds config")

    baseline: dict[str, Any] | None = None
    if args.baseline:
        baseline = try_load_json(Path(args.baseline), "baseline metrics")

    checks = run_gate(results, thresholds, baseline, strict=args.strict)

    if not checks:
        print("WARNING: No metrics found in RAGAS results — nothing to check.")
        print("         Verify that 'summary_by_group' is present in the results file.")
        sys.exit(2)

    print_report(checks)

    json_report = build_json_report(checks)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    sys.exit(0 if json_report["passed"] else 1)


if __name__ == "__main__":
    main()
