from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.instruction_following import (
    build_instruction_following_prompt,
    load_instruction_following_samples,
)


TASK_NAME = "ifeval"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_instruction_following_samples(task_dir, DATA_FILE, "IFEval")


def build_prompt(sample: Sample) -> str:
    return build_instruction_following_prompt(sample)
