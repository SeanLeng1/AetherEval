from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.eval_set_math import (
    DATA_FILE,
    build_eval_set_math_prompt,
    load_eval_set_math_samples,
)


TASK_NAME = "olympiad-bench"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_eval_set_math_samples(task_dir, DATA_FILE)


def build_prompt(sample: Sample) -> str:
    return build_eval_set_math_prompt(sample)
