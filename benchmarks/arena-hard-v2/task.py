from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "arena-hard-v2"
DATA_FILE = "data/eval.jsonl"
# Upstream utils/judge_utils.py JUDGE_SETTINGS.
BASELINE_MODELS = {
    "hard_prompt": "o3-mini-2025-01-31",
    "creative_writing": "gemini-2.0-flash-001",
}


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        category = str(row["category"])
        samples.append(
            Sample(
                id=str(row["uid"]),
                gold=None,
                data={
                    "prompt": str(row["prompt"]),
                    "baseline_answer": str(row["baseline_answer"]),
                },
                meta={
                    "category": category,
                    "subcategory": str(row.get("subcategory", "")),
                    "baseline_model": BASELINE_MODELS[category],
                    "baseline_metadata": row["baseline_metadata"],
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return str(sample.data["prompt"])
