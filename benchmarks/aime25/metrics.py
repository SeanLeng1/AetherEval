from __future__ import annotations

from collections import defaultdict
from typing import Any

from aethereval.metrics_utils import (
    mean,
    mean_stderr,
    pass_at_k,
    resolve_pass_k_values,
    to_records,
)
from aethereval.types import Sample

try:
    from math_verify.errors import TimeoutException
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "math-verify is required for AIME metrics. Install with `pip install math-verify`."
    ) from exc


_GOLD_EXTRACTION_TARGET = (LatexExtractionConfig(),)
_PRED_EXTRACTION_TARGET = (ExprExtractionConfig(), LatexExtractionConfig())


def _math_verify_reward(gold: str, prediction: str) -> tuple[float, list[str], list[str], str | None]:
    gold_text = str(gold).strip()
    gold_boxed = f"\\boxed{{{gold_text}}}"

    try:
        extracted_predictions = parse(prediction, _PRED_EXTRACTION_TARGET)
        extracted_golds = parse(gold_boxed, _GOLD_EXTRACTION_TARGET)
    except TimeoutException:
        return 0.0, [], [], "parse timeout"
    except Exception as exc:  # noqa: BLE001
        return 0.0, [], [], f"parse error: {type(exc).__name__}: {exc}"

    pred_strings = [str(x) for x in extracted_predictions]
    gold_strings = [str(x) for x in extracted_golds]

    if not extracted_golds:
        return 0.0, pred_strings, gold_strings, "no gold extraction"
    if not extracted_predictions:
        return 0.0, pred_strings, gold_strings, None

    try:
        matched = any(verify(g, p, 6) for g in extracted_golds for p in extracted_predictions)
    except TimeoutException:
        return 0.0, pred_strings, gold_strings, "verify timeout"
    except Exception as exc:  # noqa: BLE001
        return 0.0, pred_strings, gold_strings, f"verify error: {type(exc).__name__}: {exc}"

    return (1.0 if matched else 0.0), pred_strings, gold_strings, None


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    gold = str(sample.gold).strip()
    score, pred_values, gold_values, warning = _math_verify_reward(gold, generation)

    parsed = {
        "prediction_extracted": pred_values,
        "gold_extracted": gold_values,
    }
    meta: dict[str, Any] = {
        "prediction_extracted": pred_values[0] if pred_values else None,
    }
    if warning:
        meta["warning"] = warning

    return {
        "score": score,
        "is_pass": bool(score >= 1.0),
        "parsed": parsed,
        "meta": meta,
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float | list[str]]:
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

        scores = [float(r.score) for r in records]
        parsed_flags = [
            1.0
            if isinstance(r.parsed, dict)
            and isinstance(r.parsed.get("prediction_extracted"), list)
            and len(r.parsed.get("prediction_extracted")) > 0
            else 0.0
            for r in records
        ]

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

    result: dict[str, float | list[str]] = {
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
