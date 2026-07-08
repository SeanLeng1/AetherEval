from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.aime import build_aime_prompt, load_aime_samples


TASK_NAME = "aime25"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_aime_samples(task_dir, DATA_FILE)


def build_prompt(sample: Sample) -> str:
    return build_aime_prompt(sample)
