"""Convert bounded primary metrics to percentages without changing native metrics."""

import math
from typing import Any


_SCALES = {
    **dict.fromkeys(
        ("accuracy", "accuracy_first", "acc", "pass@1", "exact_match",
         "macro_accuracy", "puzzle_accuracy", "prompt_level_strict_acc",
         "prompt_level_loose_acc", "inst_level_strict_acc", "inst_level_loose_acc",
         "score"),
        100.0,
    ),
    **dict.fromkeys(
        ("overall_acc", "overall_score", "eqbench_creative_score", "coverage",
         "OP", "style_controlled_win_rate"),
        1.0,
    ),
}


def primary_score_scale(metric: str | None) -> float | None:
    # Rewards/utilities and unknown custom metrics have no implied bounded scale.
    return _SCALES.get(metric)


def primary_score_fields(
    metric: str | None, raw_score: float | None, *, scale: float | None
) -> dict[str, Any]:
    score = None
    if raw_score is not None and not math.isfinite(float(raw_score)):
        raise ValueError(f"Primary metric {metric!r} must be finite")
    if scale is not None:
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("PRIMARY_SCORE_SCALE must be a finite positive number or None")
        if raw_score is not None:
            score = float(raw_score) * scale
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f"Primary metric {metric!r} is outside its declared 0–100 scale: {score}")
    return {
        "primary_metric": metric,
        "raw_primary_score": raw_score,
        "primary_score_scale": scale,
        "primary_score": score,
    }


def normalize_task_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metric = summary.get("primary_metric")
    raw_score = summary.get(
        "raw_primary_score",
        summary.get("metrics", {}).get(metric, summary.get("primary_score")),
    )
    fields = primary_score_fields(
        metric,
        raw_score,
        scale=summary.get("primary_score_scale", primary_score_scale(metric)),
    )
    complete = summary.get("evaluation_complete")
    if complete is None:
        # BFCL snapshots record completion as generation/grading counts.
        complete = bool(summary.get("prediction_records")) and (
            summary.get("prediction_records") == summary.get("prediction_scored_records")
        )
    return {**summary, **fields, "evaluation_complete": bool(complete)}
