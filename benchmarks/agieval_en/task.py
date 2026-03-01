
from pathlib import Path
from string import ascii_uppercase

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "agieval_en"
DATA_FILE = "data/eval.jsonl"


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
        raise ValueError("AGIEval row must include at least 2 choices")
    return choice_map


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("AGIEval row must be a JSON object")

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
                    "subset": str(row.get("subset", "")).strip(),
                    "source": str(row.get("source", "")).strip(),
                },
                data={
                    "question": question,
                    "choices": choices,
                    "query": str(row.get("query", "")).strip(),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    question = str(sample.data["question"]).strip()
    choices = sample.data["choices"]
    letters = list(choices.keys())
    letter_block = ", ".join(f"({letter})" for letter in letters)
    option_lines = "\n".join(f" ({letter}) {choices[letter]}" for letter in letters)
    return (
        "Answer the following multiple-choice question by giving the correct answer letter in "
        "parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the "
        'response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one '
        f"of {letter_block}.\n\n"
        f"Question: {question}\n"
        f"{option_lines}\n\n"
        "Answer the above question and REMEMBER to finish your response with the exact phrase "
        '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of '
        f"{letter_block}."
    )
