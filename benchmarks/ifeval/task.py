from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.instruction_following import (
    build_instruction_following_prompt as build_prompt,
    load_instruction_following_samples,
)


TASK_NAME = "ifeval"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_instruction_following_samples(task_dir, DATA_FILE, "IFEval")

__all__ = ["TASK_NAME", "DATA_FILE", "load_samples", "build_prompt"]
