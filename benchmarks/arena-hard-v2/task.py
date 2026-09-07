from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "arena-hard-v2"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        if row["category"] != "hard_prompt":
            continue
        samples.append(
            Sample(
                id=str(row["uid"]),
                gold=None,
                data={
                    "prompt": str(row["prompt"]),
                    "baseline_answer": str(row["baseline_answer"]),
                },
                meta={
                    "category": "hard_prompt",
                    "subcategory": str(row.get("subcategory", "")),
                    "baseline_model": "o3-mini-2025-01-31",
                    "baseline_metadata": row["baseline_metadata"],
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return str(sample.data["prompt"])
