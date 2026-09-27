from pathlib import Path

from aethereval.core.types import Sample
from benchmark_utils.mcq import load_mcq_samples


TASK_NAME = "gpqa-diamond"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    return load_mcq_samples(
        task_dir,
        DATA_FILE,
        "GPQA",
        meta_keys=("domain", "subdomain", "record_id"),
        letters=("A", "B", "C", "D"),
    )


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
