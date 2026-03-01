
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
    "test",
    "canonical_solution",
    "base_input",
    "plus_input",
    "atol",
}


_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "You will be given a function specification and must return a correct completed "
    "Python function that passes all tests."
)
_FORMAT_INSTRUCTION = (
    "Provide a SHORT reasoning on how to solve the task, then return the completed "
    "function enclosed in a Python code block as:\n"
    "```python\n"
    "# YOUR CODE HERE\n"
    "```"
)


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
        test = str(row["test"])
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
                    "test": test,
                    "canonical_solution": canonical_solution,
                    "base_input": base_input,
                    "plus_input": plus_input,
                    "atol": atol,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    prompt = str(sample.data["prompt"])
    user_prompt = (
        f"### Question:\n{prompt}\n\n"
        "### Format:\n"
        f"{_FORMAT_INSTRUCTION}\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
