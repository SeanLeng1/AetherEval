from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample
from benchmark_utils.open_qa import normalize_answer

TASK_NAME = "qampari_oracle5"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        sample_id = str(row["id"])
        question = str(row["question"]).strip()
        passages = [
            str(value).strip() for value in row["passages"] if str(value).strip()
        ]
        answer_groups = [
            [str(alias).strip() for alias in group if str(alias).strip()]
            for group in row["answer_groups"]
        ]
        answer_groups = [group for group in answer_groups if group]
        if len(passages) != 5:
            raise ValueError(
                f"QAMPARI Oracle-5 sample {sample_id} must have five passages"
            )
        if len(answer_groups) < 5:
            raise ValueError(
                f"QAMPARI sample {sample_id} has fewer than five gold answers"
            )
        list_safe = sum(
            any(
                "," not in alias
                and "\n" not in alias
                and "\r" not in alias
                and bool(normalize_answer(alias))
                for alias in group
            )
            for group in answer_groups
        )
        if list_safe < 5:
            raise ValueError(
                f"QAMPARI Oracle-5 sample {sample_id} has fewer than five "
                "comma-safe gold answers"
            )
        samples.append(
            Sample(
                id=sample_id,
                gold=answer_groups,
                data={"question": question, "passages": passages},
                meta={"source_split": "test", "context_mode": "oracle5"},
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    passages = "\n\n".join(
        f"[{index}] {passage}"
        for index, passage in enumerate(sample.data["passages"], start=1)
    )
    return (
        "Read the passages and answer the question.\n\n"
        f"Passages:\n{passages}\n\n"
        f"Question: {sample.data['question']}\n\n"
        "Return only one comma-separated list of answer entities on a single line. "
        "Do not add an explanation or citations."
    )
