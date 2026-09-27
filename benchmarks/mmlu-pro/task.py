from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.mcq import load_mcq_samples


TASK_NAME = "mmlu-pro"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_mcq_samples(
        task_dir,
        DATA_FILE,
        "MMLU-Pro",
        meta_keys=("category", "src"),
        raw_meta_keys=("question_id",),
    )


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
