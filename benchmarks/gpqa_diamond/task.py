
from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "gpqa_diamond"
DATA_FILE = "data/eval.jsonl"


def _to_choice_map(raw: dict) -> dict[str, str]:
    keys = ("A", "B", "C", "D")
    choice_map: dict[str, str] = {}
    for key in keys:
        value = str(raw.get(key, "")).strip()
        if not value:
            raise ValueError(f"Missing choice '{key}'")
        choice_map[key] = value
    return choice_map


def load_samples(task_dir: Path) -> list[Sample]:
    data_path = task_dir / DATA_FILE
    rows = read_jsonl(data_path)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPQA row must be a JSON object")

        sample_id = str(row["id"])
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid answer label for sample {sample_id}: {answer}")

        raw_choices = row.get("choices", {})
        if not isinstance(raw_choices, dict):
            raise ValueError(f"choices must be a JSON object for sample {sample_id}")
        choices = _to_choice_map(raw_choices)

        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "domain": str(row.get("domain", "")).strip(),
                    "subdomain": str(row.get("subdomain", "")).strip(),
                    "record_id": str(row.get("record_id", "")).strip(),
                },
                data={
                    "question": question,
                    "choices": choices,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    question = str(sample.data["question"]).strip()
    choices = sample.data["choices"]
    instruction = (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        "LETTER is one of A, B, C, D. Think step by step before answering."
    )
    return (
        f"{instruction}\n\n"
        f"{question}\n\n"
        f"A) {choices['A']}\n"
        f"B) {choices['B']}\n"
        f"C) {choices['C']}\n"
        f"D) {choices['D']}"
    )
