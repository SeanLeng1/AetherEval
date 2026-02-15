from __future__ import annotations

from typing import Any

from aethereval.metrics.common import (
    aggregate_binary_results,
)
from aethereval.core.types import GenerationRecord, Sample

try:
    from math_verify.errors import TimeoutException
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "math-verify is required for AIME metrics. Install with `pip install math-verify`."
    ) from exc


PRIMARY_METRIC = "accuracy"


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


def _parsed_prediction_extracted(record: GenerationRecord) -> bool:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    extracted = parsed.get("prediction_extracted")
    return isinstance(extracted, list) and len(extracted) > 0


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float | list[str]]:
    return aggregate_binary_results(
        sample_results,
        metric_options,
        parsed_flag_fn=_parsed_prediction_extracted,
    )
