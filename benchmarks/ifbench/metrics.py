from typing import Any

from benchmarks.ifbench.ifbench_lib import evaluation_lib
from benchmark_utils.instruction_following import (
    aggregate_instruction_following,
    score_instruction_following,
)
from aethereval.core.types import Sample


PRIMARY_METRIC = "prompt_level_loose_acc"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    return score_instruction_following(sample, generation, evaluation_lib)


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_instruction_following(sample_results, metric_options)
