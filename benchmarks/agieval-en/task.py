from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.mcq import load_mcq_samples


TASK_NAME = "agieval-en"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_mcq_samples(
        task_dir,
        DATA_FILE,
        "AGIEval",
        meta_keys=("subset", "source"),
        data_keys=("query",),
    )


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
