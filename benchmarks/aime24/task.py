from __future__ import annotations

from pathlib import Path

from aethereval.io import read_jsonl
from aethereval.types import Sample


TASK_NAME = "aime24"
DATA_FILE = "data/eval.jsonl"
DEFAULT_GEN = {
    "n": 16,
    "max_new_tokens": 4096,
    "temperature": 0.7,
    "top_p": 1.0,
}

_MATH_PROMPT_TEMPLATE = (
    "{Question}\n\n"
    "Please think step by step, and put your final answer within \\boxed{{}}."
)


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("AIME row must be a JSON object")
        sample_id = str(row["id"])
        problem = str(row["problem"]).strip()
        answer = str(row["answer"]).strip()
        if not problem:
            raise ValueError(f"Empty problem for sample {sample_id}")
        if not answer:
            raise ValueError(f"Empty answer for sample {sample_id}")
        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "year": row.get("year"),
                    "url": row.get("url"),
                },
                data={
                    "problem": problem,
                    "solution": row.get("solution"),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return _MATH_PROMPT_TEMPLATE.format(Question=str(sample.data["problem"]))
