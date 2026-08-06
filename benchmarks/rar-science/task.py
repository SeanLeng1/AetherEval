from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.rar import build_rar_prompt, load_rar_samples


TASK_NAME = "rar-science"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_rar_samples(task_dir, DATA_FILE, expected_domain="Science")


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    return build_rar_prompt(sample)
