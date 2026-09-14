from typing import Any

from aethereval.core.task_defaults import resolve_task_default_metrics
from aethereval.core.types import GenerationOutput, Sample
from benchmark_utils.rar import (
    aggregate_rar as aggregate,
    score_rar_generations_batch,
    validate_rar_metric_options,
)


PRIMARY_METRIC = "score"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("rar-science").get(
        "judge_model", "google/gemma-4-26B-A4B-it"
    )
)


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    validate_rar_metric_options(metric_options, default_model=DEFAULT_JUDGE_MODEL)


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    return score_rar_generations_batch(
        samples,
        generation_outputs,
        metric_options,
        default_model=DEFAULT_JUDGE_MODEL,
    )

__all__ = ["PRIMARY_METRIC", "score_generations_batch", "aggregate", "validate_metric_options"]
