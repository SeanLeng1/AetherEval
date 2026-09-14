from benchmark_utils.eval_set_math import (
    DATA_FILE,
    build_eval_set_math_prompt as build_prompt,
    load_eval_set_math_samples as load_samples,
)

TASK_NAME = "math500"

__all__ = ["TASK_NAME", "DATA_FILE", "load_samples", "build_prompt"]
