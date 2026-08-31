from typing import Any

from aethereval.core.types import Sample
from benchmark_utils.open_qa import aggregate_qampari, score_qampari

PRIMARY_METRIC = "qampari_f1_top5"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    return score_qampari(sample, generation)


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_qampari(sample_results, metric_options)
