from typing import Any

from benchmarks.ifbench.ifbench_lib import evaluation_lib
from benchmark_utils.instruction_following import (
    aggregate_instruction_following as aggregate,
    score_instruction_following,
)
from aethereval.core.types import Sample


PRIMARY_METRIC = "prompt_level_loose_acc"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    return score_instruction_following(sample, generation, evaluation_lib)

__all__ = ["PRIMARY_METRIC", "score_generation", "aggregate"]
