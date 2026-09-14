from benchmark_utils.open_qa import (
    aggregate_open_qa as aggregate,
    score_open_qa as score_generation,
)

PRIMARY_METRIC = "exact_match"

__all__ = ["PRIMARY_METRIC", "aggregate", "score_generation"]
