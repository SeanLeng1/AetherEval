from __future__ import annotations

from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "humaneval_plus"
DATA_FILE = "data/eval.jsonl"


_REQUIRED_KEYS = {
    "task_id",
    "prompt",
    "entry_point",
    "canonical_solution",
    "base_input",
    "plus_input",
    "atol",
}


def _ensure_list(value: Any, key: str, sample_id: str) -> list[Any]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{key} must be list for sample {sample_id}")


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("HumanEval+ row must be a JSON object")

        missing = sorted(_REQUIRED_KEYS - set(row.keys()))
        if missing:
            raise ValueError(f"HumanEval+ row missing keys: {', '.join(missing)}")

        task_id = str(row["task_id"]).strip()
        if not task_id:
            raise ValueError("task_id must be non-empty")

        prompt = str(row["prompt"])
        entry_point = str(row["entry_point"]).strip()
        canonical_solution = str(row["canonical_solution"])
        contract = str(row.get("contract", ""))
        base_input = _ensure_list(row["base_input"], "base_input", task_id)
        plus_input = _ensure_list(row["plus_input"], "plus_input", task_id)
        atol = float(row["atol"])

        samples.append(
            Sample(
                id=task_id,
                gold=None,
                meta={
                    "entry_point": entry_point,
                    "source": "evalplus/HumanEvalPlus",
                },
                data={
                    "task_id": task_id,
                    "prompt": prompt,
                    "contract": contract,
                    "entry_point": entry_point,
                    "canonical_solution": canonical_solution,
                    "base_input": base_input,
                    "plus_input": plus_input,
                    "atol": atol,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    prompt = str(sample.data["prompt"])
    contract = str(sample.data.get("contract", "")).strip()
    if contract:
        return (
            "Complete the Python function below. "
            "Return only executable Python code (no explanation).\n\n"
            "# Function signature and docstring\n"
            f"{prompt}\n"
            "# Input contract hints (optional assertions)\n"
            f"{contract}\n"
        )
    return (
        "Complete the Python function below. "
        "Return only executable Python code (no explanation).\n\n"
        f"{prompt}\n"
    )
