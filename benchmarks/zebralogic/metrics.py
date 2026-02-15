from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from aethereval.metrics.common import mean, mean_stderr, pass_at_k, resolve_pass_k_values, to_records
from aethereval.core.types import Sample


PRIMARY_METRIC = "accuracy"


_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:：]\s*(?P<ans>.+?)\s*$")


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _extract_answer(generation: str) -> tuple[str, str]:
    matches = list(_ANSWER_LINE_RE.finditer(generation))
    if matches:
        value = matches[-1].group("ans").strip().strip("` ")
        return value, "answer_line"

    lines = [line.strip() for line in generation.splitlines() if line.strip()]
    if lines:
        return lines[-1].strip("` "), "last_line"

    return "", "empty"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    pred_raw, method = _extract_answer(generation)
    pred = _normalize(pred_raw)
    gold = _normalize(str(sample.gold))
    score = 1.0 if pred == gold else 0.0

    parsed = {
        "prediction": pred_raw,
        "prediction_norm": pred,
        "gold_norm": gold,
        "extract_method": method,
    }
    return {
        "score": score,
        "is_pass": bool(score >= 1.0),
        "parsed": parsed,
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

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        scores: list[float] = []
        parsed_flags: list[float] = []
        for r in records:
            scores.append(float(r.score))
            parsed = r.parsed if isinstance(r.parsed, dict) else {}
            prediction = parsed.get("prediction")
            parsed_flags.append(1.0 if isinstance(prediction, str) and prediction.strip() else 0.0)

        sample_mean_scores.append(mean(scores))
        sample_parsed_rates.append(mean(parsed_flags))
        sample_binary_scores.append([1 if s >= 1.0 else 0 for s in scores])

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

    return result
