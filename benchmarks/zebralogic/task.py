from __future__ import annotations

from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "zebralogic"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ZebraLogic row must be a JSON object")

        sample_id = str(row["id"])
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        if not question:
            raise ValueError(f"Empty question for sample {sample_id}")
        if not answer:
            raise ValueError(f"Empty answer for sample {sample_id}")

        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "subset": str(row.get("subset", "")).strip(),
                    "source": str(row.get("source", "")).strip(),
                },
                data={
                    "question": question,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    question = str(sample.data["question"])
    return (
        "Solve the following logic puzzle. "
        "Return only the final answer.\n"
        "Use the last line format: ANSWER: <answer>\n\n"
        f"Question:\n{question}\n\nAnswer:"
    )
