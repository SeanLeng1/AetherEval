from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import (
    GenerationInput,
    GenerationOutput,
    GenerationRecord,
    Sample,
)


TASK_NAME = "creative_writing_v3"
DATA_FILE = "data/eval.jsonl"
DEFAULT_GEN = {
    "n": 1,
    "max_new_tokens": 12000,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.1,
}


def load_samples(task_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / DATA_FILE):
        sample_id = str(row["id"])
        base_prompt = str(row["base_prompt"])
        seed_modifier = str(row["seed_modifier"])
        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                data={
                    "base_prompt": base_prompt,
                    "seed_modifier": seed_modifier,
                    "prompt": base_prompt.replace("<SEED>", seed_modifier),
                },
                meta={
                    "prompt_id": str(row["prompt_id"]),
                    "iteration": int(row["iteration"]),
                    "category": str(row.get("category", "")),
                    "title": str(row.get("title", "")),
                },
            )
        )
    return samples


def build_prompt(sample: Sample) -> str:
    return str(sample.data["prompt"])


def generate_outputs(
    *,
    backend: Any,
    samples: list[Sample],
    pending_indices: dict[str, list[int]],
    existing_records: list[GenerationRecord],
    gen_cfg: dict[str, Any],
) -> list[GenerationOutput]:
    del existing_records
    if int(gen_cfg["n"]) != 1:
        raise ValueError("creative_writing_v3 requires n=1; iterations are data rows")

    sample_by_id = {sample.id: sample for sample in samples}
    pending_ids = [sample.id for sample in samples if pending_indices.get(sample.id)]
    remaining = list(pending_ids)
    selected: dict[str, GenerationOutput] = {}

    for attempt in range(3):
        inputs = [
            GenerationInput(
                sample_id=sample_id,
                prompt=[{"role": "user", "content": build_prompt(sample_by_id[sample_id])}],
                num_generations=1,
            )
            for sample_id in remaining
        ]
        outputs = backend.generate(inputs, gen_cfg)
        by_id = {output.sample_id: output for output in outputs}
        if set(by_id) != set(remaining):
            raise ValueError("creative_writing_v3 backend output ids mismatch")

        retry: list[str] = []
        for sample_id in remaining:
            output = by_id[sample_id]
            if len(output.generations) != 1:
                raise ValueError("creative_writing_v3 expects one generation per sample")
            text = output.generations[0].strip()
            failed = output.error is not None or len(text) < 500
            if failed and attempt < 2:
                retry.append(sample_id)
                continue
            meta = dict(output.meta)
            # The upstream runner stores response.strip(); force token counting
            # against the normalized text rather than backend metadata for the
            # pre-strip response.
            meta.pop("response_token_counts", None)
            meta["creative_generation_failed"] = bool(failed)
            if output.error is not None:
                meta["creative_generation_error"] = str(output.error)
            selected[sample_id] = GenerationOutput(
                sample_id=output.sample_id,
                prompt=output.prompt,
                generations=[text],
                # Upstream excludes a piece after three failed attempts instead
                # of aborting the entire benchmark run.
                error=None,
                meta=meta,
            )
        remaining = retry
        if not remaining:
            break

    return [selected[sample_id] for sample_id in pending_ids]
