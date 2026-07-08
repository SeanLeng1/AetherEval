from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "safe_alignment"
DATA_FILE = "data/eval.jsonl"


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)
    samples: list[Sample] = []
    for row in rows:
        sample_id = str(row["id"])
        messages = _normalize_messages(row["prompt"], sample_id)
        data_source = str(row["data_source"]).strip()
        ability = str(row["ability"]).strip()
        if not data_source:
            raise ValueError(f"Empty data_source for sample {sample_id}")
        if not ability:
            raise ValueError(f"Empty ability for sample {sample_id}")

        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                data={
                    "prompt": messages,
                    "data_source": data_source,
                    "ability": ability,
                    "reward_model": row["reward_model"],
                    "extra_info": row["extra_info"],
                },
                meta={
                    "data_source": data_source,
                    "ability": ability,
                    "source": str(row["source"]),
                    "source_file": str(row["source_file"]),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    return _normalize_messages(sample.data["prompt"], sample.id)


def _normalize_messages(raw: Any, sample_id: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompt must be a non-empty list for sample {sample_id}")

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"prompt[{idx}] must be an object for sample {sample_id}")
        role = str(item["role"]).strip()
        content = str(item["content"])
        if not role:
            raise ValueError(f"prompt[{idx}].role is empty for sample {sample_id}")
        messages.append({"role": role, "content": content})
    return messages
