from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "healthbench"
DATA_FILE = "data/eval.jsonl"
DEFAULT_GEN = {
    "n": 1,
    "max_new_tokens": 2048,
    "temperature": 0.5,
    "top_p": 1.0,
}
SYSTEM_MESSAGE = "You are a helpful assistant."


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for idx, row in enumerate(read_jsonl(task_dir / DATA_FILE)):
        sample_id = str(row.get("id") or row.get("prompt_id") or idx)
        prompt = _messages(row["prompt"], sample_id)
        rubrics = row["rubrics"]
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError(f"HealthBench sample {sample_id} has no rubrics")
        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                data={
                    "prompt": prompt,
                    "rubrics": rubrics,
                    "example_tags": list(row.get("example_tags", [])),
                },
                meta={
                    "prompt_id": str(row.get("prompt_id", sample_id)),
                    "example_tags": list(row.get("example_tags", [])),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    # simple-evals' ChatCompletionSampler prepends this system message to the
    # candidate request; keep the raw conversation unchanged for the grader.
    return [{"role": "system", "content": SYSTEM_MESSAGE}] + _messages(
        sample.data["prompt"], sample.id
    )


def _messages(raw: Any, sample_id: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"HealthBench prompt must be non-empty for {sample_id}")
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid HealthBench message for {sample_id}")
        messages.append({"role": str(item["role"]), "content": str(item["content"])})
    return messages
