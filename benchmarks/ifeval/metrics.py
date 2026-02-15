from __future__ import annotations

from typing import Any

from aethereval.metrics.common import aggregate_instruction_following_results
from aethereval.core.types import Sample
from benchmarks.ifeval.ifeval_lib import evaluation_lib


PRIMARY_METRIC = "prompt_level_strict_acc"


def _build_input_example(sample: Sample) -> evaluation_lib.InputExample:
    prompt = str(sample.data.get("prompt", ""))
    instruction_id_list = [str(x) for x in sample.meta.get("instruction_id_list", [])]

    raw_kwargs = list(sample.meta.get("kwargs", []))
    kwargs: list[dict[str, Any]] = []
    for idx in range(len(instruction_id_list)):
        item = raw_kwargs[idx] if idx < len(raw_kwargs) else {}
        if not isinstance(item, dict):
            item = {}
        kwargs.append({k: v for k, v in item.items() if v is not None})

    try:
        key: Any = int(sample.id)
    except Exception:  # noqa: BLE001
        key = sample.id

    return evaluation_lib.InputExample(
        key=key,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=kwargs,
    )


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    inp = _build_input_example(sample)
    prompt_to_response = {inp.prompt: generation}

    strict_out = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    loose_out = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)

    prompt_level_strict_acc = float(bool(strict_out.follow_all_instructions))
    prompt_level_loose_acc = float(bool(loose_out.follow_all_instructions))

    parsed = {
        "prompt_level_strict_acc": prompt_level_strict_acc,
        "inst_level_strict_acc": [bool(x) for x in strict_out.follow_instruction_list],
        "prompt_level_loose_acc": prompt_level_loose_acc,
        "inst_level_loose_acc": [bool(x) for x in loose_out.follow_instruction_list],
    }
    return {
        "score": prompt_level_strict_acc,
        "is_pass": bool(prompt_level_strict_acc),
        "parsed": parsed,
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float | list[str]]:
    del metric_options
    return aggregate_instruction_following_results(sample_results)
