from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import (
    GenerationInput,
    GenerationOutput,
    GenerationRecord,
    Sample,
)


TASK_NAME = "llmeval-med"
DATA_FILE = "data/eval.jsonl"
SYSTEM_MESSAGE = "You are a helpful assistant."


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        sample_id = str(row["id"])
        samples.append(
            Sample(
                id=sample_id,
                gold=row.get("sanswer"),
                data={
                    "problem": str(row["problem"]),
                    "sanswer": row.get("sanswer"),
                    "checklist": row.get("checklist"),
                    "category": str(row["category"]),
                    "group_code": str(row["groupCode"]),
                    "round": int(row["round"]),
                },
                meta={
                    "category": str(row["category"]),
                    "category2": str(row.get("category2", "")),
                    "scene": str(row.get("scene", "")),
                    "difficulty": str(row.get("difficulty", "")),
                    "group_code": str(row["groupCode"]),
                    "round": int(row["round"]),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": str(sample.data["problem"])},
    ]


def generate_outputs(
    *,
    backend: Any,
    samples: list[Sample],
    pending_indices: dict[str, list[int]],
    existing_records: list[GenerationRecord],
    gen_cfg: dict[str, Any],
) -> list[GenerationOutput]:
    if int(gen_cfg["n"]) != 1:
        raise ValueError("llmeval_med requires n=1 because later turns depend on turn 1")

    existing = {
        record.sample_id: record for record in existing_records if record.gen_idx == 0
    }
    histories: dict[tuple[str, str], list[tuple[str, str]]] = {}
    generated: dict[str, GenerationOutput] = {}
    max_round = max((int(sample.data["round"]) for sample in samples), default=0)

    for round_idx in range(1, max_round + 1):
        current = [
            sample for sample in samples if int(sample.data["round"]) == round_idx
        ]
        inputs: list[GenerationInput] = []
        for sample in current:
            if not pending_indices.get(sample.id):
                continue
            key = (str(sample.data["category"]), str(sample.data["group_code"]))
            messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
            for question, answer in histories.get(key, []):
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": str(sample.data["problem"])})
            inputs.append(
                GenerationInput(
                    sample_id=sample.id,
                    prompt=messages,
                    num_generations=1,
                )
            )

        if inputs:
            outputs = backend.generate(inputs, gen_cfg)
            returned = {output.sample_id: output for output in outputs}
            if set(returned) != {item.sample_id for item in inputs}:
                raise ValueError("llmeval_med backend output ids mismatch")
            for sample_id, output in returned.items():
                if len(output.generations) != 1:
                    raise ValueError("llmeval_med expects one generation per turn")
                meta = dict(output.meta)
                meta.pop("response_token_counts", None)
                generated[sample_id] = GenerationOutput(
                    sample_id=output.sample_id,
                    prompt=output.prompt,
                    generations=[output.generations[0].strip()],
                    error=output.error,
                    meta=meta,
                )

        for sample in current:
            key = (str(sample.data["category"]), str(sample.data["group_code"]))
            if sample.id in generated:
                output = generated[sample.id]
                if len(output.generations) != 1:
                    raise ValueError("llmeval_med expects one generation per turn")
                answer = output.generations[0]
            elif sample.id in existing:
                answer = existing[sample.id].generation
            else:
                raise ValueError(f"Missing LLMEval-Med response for {sample.id}")
            histories.setdefault(key, []).append(
                (str(sample.data["problem"]), answer)
            )

    return [sample_output for sample in samples if (sample_output := generated.get(sample.id))]
