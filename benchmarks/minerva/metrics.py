from typing import Any

from aethereval.core.types import Sample
from benchmark_utils.eval_set_math import aggregate
from benchmark_utils.eval_set_math import score_generation as _score_eval_set_math


PRIMARY_METRIC = "accuracy"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    # Minerva golds end in physics variables (t, m, c) that math-verify would
    # strip from predictions as units; see benchmark_utils/math_scoring.py.
    return _score_eval_set_math(sample, generation, keep_units_fallback=True)


__all__ = ["PRIMARY_METRIC", "aggregate", "score_generation"]
