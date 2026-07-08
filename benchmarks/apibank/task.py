from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "apibank"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    samples: list[Sample] = []
    for row in rows:
        sample_id = str(row["id"]).strip()
        if not sample_id:
            raise ValueError("APIBank sample id is empty")

        level = int(row["level"])
        if level not in (1, 2, 3):
            raise ValueError(f"Invalid APIBank level for sample {sample_id}: {level}")
        if not sample_id.startswith(f"Level{level}_"):
            raise ValueError(
                f"APIBank sample id/level mismatch: id={sample_id} level={level}"
            )

        system = str(row["system"])
        user = str(row["user"])
        if not system.strip():
            raise ValueError(f"Empty system prompt for sample {sample_id}")
        if not user.strip():
            raise ValueError(f"Empty user prompt for sample {sample_id}")

        _validate_answer(row["answer"], sample_id)
        samples.append(
            Sample(
                id=sample_id,
                gold=row["answer"],
                meta={
                    "level": level,
                    "source_index": int(row["source_index"]),
                    "source": str(row["source"]),
                },
                data={
                    "system": system,
                    "user": user,
                    "answer": row["answer"],
                    "other": row["other"],
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": str(sample.data["system"])},
        {"role": "user", "content": str(sample.data["user"])},
    ]


def _validate_answer(answer: object, sample_id: str) -> None:
    if isinstance(answer, list):
        if not answer:
            raise ValueError(f"Empty answer list for sample {sample_id}")
        answer = answer[0]
    if not isinstance(answer, dict):
        raise ValueError(f"Answer must be an object for sample {sample_id}")
    if not isinstance(answer["name"], str) or not answer["name"].strip():
        raise ValueError(
            f"Answer name must be a non-empty string for sample {sample_id}"
        )
    if not isinstance(answer["parameters"], dict):
        raise ValueError(f"Answer parameters must be an object for sample {sample_id}")
