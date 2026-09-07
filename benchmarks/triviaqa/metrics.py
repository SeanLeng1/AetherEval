from typing import Any

from aethereval.core.types import Sample
from benchmark_utils.open_qa import aggregate_open_qa, score_open_qa

PRIMARY_METRIC = "exact_match"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    return score_open_qa(sample, generation)


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_open_qa(sample_results, metric_options)
