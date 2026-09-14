from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.rar import build_rar_prompt as build_prompt, load_rar_samples


TASK_NAME = "rar-medical"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_rar_samples(task_dir, DATA_FILE, expected_domain="Medical")

__all__ = ["TASK_NAME", "DATA_FILE", "load_samples", "build_prompt"]
