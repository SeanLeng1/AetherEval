from __future__ import annotations

from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "ifbench"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("IFBench row must be a JSON object")

        sample_id = str(row["key"])
        prompt = str(row["prompt"])
        instruction_id_list = row.get("instruction_id_list", [])
        kwargs = row.get("kwargs", [])
        if not isinstance(instruction_id_list, list):
            raise ValueError(f"instruction_id_list must be list for sample {sample_id}")
        if not isinstance(kwargs, list):
            raise ValueError(f"kwargs must be list for sample {sample_id}")

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
