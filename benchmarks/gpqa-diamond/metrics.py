from typing import Any

from aethereval.metrics.common import (
    aggregate_mcq_results,
    score_generation_mcq as score_generation,
)


PRIMARY_METRIC = "accuracy"


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float | list[str]]:
    return aggregate_mcq_results(sample_results, metric_options, group_key="domain")

__all__ = ["PRIMARY_METRIC", "score_generation", "aggregate"]
