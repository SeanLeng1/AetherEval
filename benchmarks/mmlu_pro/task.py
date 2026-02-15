from __future__ import annotations

from pathlib import Path
from string import ascii_uppercase

from aethereval.io import read_jsonl
from aethereval.types import Sample


TASK_NAME = "mmlu_pro"
DATA_FILE = "data/eval.jsonl"
DEFAULT_GEN = {
    "n": 1,
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
}


def _to_choice_map(raw: dict) -> dict[str, str]:
    choice_map: dict[str, str] = {}
    for letter in ascii_uppercase:
        if letter not in raw:
            break
        value = str(raw.get(letter, "")).strip()
        if not value:
            raise ValueError(f"Missing choice '{letter}'")
        choice_map[letter] = value
    if len(choice_map) < 2:
        raise ValueError("MMLU-Pro item must include at least 2 choices")
    return choice_map


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("MMLU-Pro row must be a JSON object")

        sample_id = str(row["id"])
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip().upper()
        if not question:
            raise ValueError(f"Empty question for sample {sample_id}")

        raw_choices = row.get("choices", {})
        if not isinstance(raw_choices, dict):
            raise ValueError(f"choices must be a JSON object for sample {sample_id}")
        choices = _to_choice_map(raw_choices)

        if answer not in choices:
            raise ValueError(f"Invalid answer label for sample {sample_id}: {answer}")

        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "category": str(row.get("category", "")).strip(),
                    "src": str(row.get("src", "")).strip(),
                    "question_id": row.get("question_id"),
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
    letters = list(choices.keys())
    letters_str = ", ".join(letters)
    option_lines = "\n".join(f"{letter}) {choices[letter]}" for letter in letters)

    instruction = (
        "Answer the following multiple choice question. The last line of your response "
        "should be of the following format: 'Answer: $LETTER' (without quotes) where "
        f"LETTER is one of {letters_str}. Think step by step before answering."
    )
    return f"{instruction}\n\n{question}\n\n{option_lines}"
