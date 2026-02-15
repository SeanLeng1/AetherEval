from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aethereval.metrics_utils import mean, mean_stderr, pass_at_k, resolve_pass_k_values, to_records
from aethereval.types import Sample


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from lcb_eval_runtime import evaluate_candidate


_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[A-Za-z0-9_-]*)\s*\n(?P<code>.*?)```",
    re.DOTALL,
)


def _extract_code(text: str) -> tuple[str, str]:
    matches = list(_CODE_BLOCK_RE.finditer(text))
    if matches:
        python_blocks = []
        for match in matches:
            lang = match.group("lang").strip().lower()
            code = match.group("code").rstrip()
            if lang in {"python", "py"}:
                python_blocks.append(code)
        if python_blocks:
            return python_blocks[-1], "fenced_python"
        return matches[-1].group("code").rstrip(), "fenced_code"

    stripped = text.strip()
    if stripped:
        return stripped, "raw_text"
    return "", "empty"


def _first_error_code(statuses: list[int | bool]) -> int | None:
    for value in statuses:
        if isinstance(value, bool):
            continue
        return int(value)
    return None


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    code, extract_method = _extract_code(generation)
    had_code = bool(code.strip())
    if not had_code:
        return {
            "score": 0.0,
            "is_pass": False,
            "parsed": {
                "extract_method": extract_method,
                "had_code": False,
                "num_tests": len(sample.data.get("outputs", [])),
                "passed_tests": 0,
                "statuses": [],
            },
            "meta": {"error": "empty_generation"},
        }

    timeout = int(sample.data.get("timeout_sec", 6))
    inputs = list(sample.data.get("inputs", []))
    outputs = list(sample.data.get("outputs", []))
    fn_name = sample.data.get("fn_name")
    fn_name = str(fn_name) if fn_name is not None else None

    statuses, runtime_error = evaluate_candidate(
        code=code,
        inputs=inputs,
        outputs=outputs,
        fn_name=fn_name,
        timeout=timeout,
    )
    passed = bool(statuses) and all(value is True for value in statuses)
    passed_tests = sum(1 for value in statuses if value is True)

    parsed = {
        "extract_method": extract_method,
        "had_code": True,
        "num_tests": len(statuses),
        "passed_tests": passed_tests,
        "statuses": statuses,
    }
    meta: dict[str, Any] = {
        "error_code": _first_error_code(statuses),
    }
    if runtime_error:
        meta["runtime_error"] = runtime_error

    return {
        "score": 1.0 if passed else 0.0,
        "is_pass": passed,
        "parsed": parsed,
        "meta": meta,
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    options = metric_options or {}
    n_hint = int(options.get("n", 0)) if options.get("n") is not None else 0

    if not sample_results:
        return {
            "accuracy": 0.0,
            "accuracy_stderr": 0.0,
            "parsed_rate": 0.0,
            "pass@1": 0.0,
            "pass@1_stderr": 0.0,
        }

    sample_mean_scores: list[float] = []
    sample_parsed_rates: list[float] = []
    sample_binary_scores: list[list[int]] = []
    grouped_scores: dict[str, list[float]] = defaultdict(list)

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        scores: list[float] = []
        parsed_flags: list[float] = []
        for record in records:
            scores.append(float(record.score))
            parsed = record.parsed if isinstance(record.parsed, dict) else {}
            parsed_flags.append(1.0 if bool(parsed.get("had_code", False)) else 0.0)

        sample_mean = mean(scores)
        sample_mean_scores.append(sample_mean)
        sample_parsed_rates.append(mean(parsed_flags))
        sample_binary_scores.append([1 if s >= 1.0 else 0 for s in scores])

        meta = item.get("meta", {}) if isinstance(item.get("meta", {}), dict) else {}
        platform = str(meta.get("platform", "")).strip().lower().replace("-", "_")
        platform = re.sub(r"[^a-z0-9_]+", "_", platform).strip("_")
        if platform:
            grouped_scores[platform].append(sample_mean)

    if not sample_mean_scores:
        return {
            "accuracy": 0.0,
            "accuracy_stderr": 0.0,
            "parsed_rate": 0.0,
            "pass@1": 0.0,
            "pass@1_stderr": 0.0,
        }

    n_ref = n_hint if n_hint > 0 else max(len(x) for x in sample_binary_scores)
    pass_k_values = resolve_pass_k_values(options.get("pass_k_values"), n_ref)
    if 1 not in pass_k_values and n_ref >= 1:
        pass_k_values = [1] + pass_k_values

    pass_metrics: dict[int, list[float]] = defaultdict(list)
    for binary_scores in sample_binary_scores:
        for k in pass_k_values:
            if k <= len(binary_scores):
                pass_metrics[k].append(pass_at_k(binary_scores, k))

    result: dict[str, float] = {
        "accuracy": mean(sample_mean_scores),
        "accuracy_stderr": mean_stderr(sample_mean_scores),
        "parsed_rate": mean(sample_parsed_rates),
    }
    if n_ref > 1:
        result[f"accuracy@{n_ref}"] = result["accuracy"]

    for k in sorted(pass_metrics):
        values = pass_metrics[k]
        result[f"pass@{k}"] = mean(values)
        result[f"pass@{k}_stderr"] = mean_stderr(values)

    for platform in sorted(grouped_scores):
        result[f"accuracy_{platform}"] = mean(grouped_scores[platform])

    return result
