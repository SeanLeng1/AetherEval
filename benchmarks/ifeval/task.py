from __future__ import annotations

from pathlib import Path

from aethereval.io import read_jsonl
from aethereval.types import Sample


TASK_NAME = "ifeval"
DATA_FILE = "data/eval.jsonl"
DEFAULT_GEN = {
    "n": 1,
    "max_new_tokens": 1280,
    "temperature": 0.0,
    "top_p": 1.0,
}


def load_samples(task_dir: Path) -> list[Sample]:
    data_path = task_dir / DATA_FILE
    rows = read_jsonl(data_path)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("IFEval row must be a JSON object")

        sample_id = str(row["key"])
        prompt = str(row["prompt"])
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs = row.get("kwargs", [])

        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                meta={
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs,
                },
                data={"prompt": prompt},
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    return str(sample.data["prompt"])
