from typing import Any

from aethereval.metrics.common import aggregate_binary_results
from benchmark_utils.mcq import score_generation_mcq as score_generation


PRIMARY_METRIC = "accuracy"


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_binary_results(sample_results, metric_options, group_key="category")

__all__ = ["PRIMARY_METRIC", "score_generation", "aggregate"]
