from typing import Any

from aethereval.core.types import Sample
from benchmark_utils.eval_set_math import aggregate
from benchmark_utils.eval_set_math import score_generation as _score_eval_set_math


PRIMARY_METRIC = "accuracy"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    # AIME gold is a bare answer, so it is boxed before math-verify extracts it.
    return _score_eval_set_math(sample, generation, boxed_gold=True)


__all__ = ["PRIMARY_METRIC", "aggregate", "score_generation"]
