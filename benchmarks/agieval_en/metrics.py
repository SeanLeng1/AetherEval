from __future__ import annotations

from typing import Any

from aethereval.metrics.common import aggregate_mcq_results, score_generation_mcq
from aethereval.core.types import Sample


PRIMARY_METRIC = "accuracy"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    return score_generation_mcq(sample, generation)


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_mcq_results(sample_results, metric_options, group_key="subset")
