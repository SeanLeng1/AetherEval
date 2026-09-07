from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample

TASK_NAME = "nq-open"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for index, row in enumerate(read_jsonl(task_dir / DATA_FILE)):
        question = str(row["question"]).strip()
        answers = [
            str(answer).strip() for answer in row["answers"] if str(answer).strip()
        ]
        if not answers:
            raise ValueError(f"NQ-Open sample {index} has no answers")
        samples.append(
            Sample(
                id=str(row.get("id", index)),
                gold=answers,
                data={"question": question},
                meta={"source_split": "dev"},
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return (
        "Answer the following question with only the short answer. "
        "Do not include an explanation.\n\n"
        f"Question: {sample.data['question']}\n"
        "Answer:"
    )
