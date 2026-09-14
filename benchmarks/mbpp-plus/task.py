from pathlib import Path

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "mbpp-plus"
DATA_FILE = "data/eval.jsonl"
DATA_VERSION = "v0.2.0"


def load_samples(task_dir: Path) -> list[Sample]:
    samples = []
    seen = set()
    for row in read_jsonl(task_dir / DATA_FILE):
        for key in ("task_id", "prompt", "entry_point", "canonical_solution",
                    "base_input", "plus_input", "atol"):
            if key not in row:
                raise ValueError(f"MBPP+ row missing {key}")
        task_id = row["task_id"]
        if task_id in seen:
            raise ValueError(f"Duplicate MBPP+ task: {task_id}")
        seen.add(task_id)
        for split in ("base_input", "plus_input"):
            # The pinned release represents Mbpp/793's empty Plus suite as {}.
            if not isinstance(row[split], list) and row[split] != {}:
                raise ValueError(f"{task_id}: {split} must be a list")
        # Keep the serialized inputs in Sample: records must remain JSON-safe.
        samples.append(Sample(
            id=task_id,
            meta={"source": "evalplus/MBPPPlus", "source_version": DATA_VERSION},
            data=row,
        ))
    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    # Match our HumanEval+ reasoning-and-code format; no assistant/code prefill.
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. "
                "You will be given a function specification and must return a correct completed "
                "Python function that passes all tests."
            ),
        },
        {
            "role": "user",
            "content": (
                f"### Question:\n{sample.data['prompt']}\n\n"
                "### Format:\n"
                "Provide a SHORT reasoning on how to solve the task, then return the completed "
                "function enclosed in a Python code block as:\n"
                "```python\n# YOUR CODE HERE\n```\n\n"
                "### Answer: (use the provided format with backticks)\n\n"
            ),
        },
    ]
