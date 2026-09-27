from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample
from aethereval.metrics.common import mean, mean_stderr, to_records


def load_instruction_following_samples(
    task_dir: Path, data_file: str, benchmark_name: str
) -> list[Sample]:
    rows = read_jsonl(task_dir / data_file)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{benchmark_name} row must be a JSON object")

        sample_id = str(row["key"])
        prompt = str(row["prompt"])
        instruction_id_list = row["instruction_id_list"]
        kwargs = row["kwargs"]
        if not isinstance(instruction_id_list, list):
            raise ValueError(f"instruction_id_list must be list for sample {sample_id}")
        if not instruction_id_list:
            raise ValueError(f"instruction_id_list is empty for sample {sample_id}")
        if not isinstance(kwargs, list):
            raise ValueError(f"kwargs must be list for sample {sample_id}")

        samples.append(
            Sample(
                id=sample_id,
                gold=None,
                meta={
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs,
                },
                data={"prompt": prompt},
            )
        )
    return samples


def build_instruction_following_prompt(sample: Sample) -> str:
    return str(sample.data["prompt"])


def build_input_example(sample: Sample, evaluation_lib: Any) -> Any:
    prompt = str(sample.data["prompt"])
    instruction_id_list = [str(x) for x in sample.meta["instruction_id_list"]]

    raw_kwargs = sample.meta["kwargs"]
    if not isinstance(raw_kwargs, list):
        raise ValueError(f"kwargs must be list for sample {sample.id}")

    kwargs: list[dict[str, Any]] = []
    for idx in range(len(instruction_id_list)):
        item = raw_kwargs[idx] if idx < len(raw_kwargs) else {}
        if not isinstance(item, dict):
            raise ValueError(f"kwargs[{idx}] must be dict for sample {sample.id}")
        kwargs.append({k: v for k, v in item.items() if v is not None})

    try:
        key: Any = int(sample.id)
    except ValueError:
        key = sample.id

    return evaluation_lib.InputExample(
        key=key,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=kwargs,
    )


def score_instruction_following(
    sample: Sample, generation: str, evaluation_lib: Any
) -> dict[str, Any]:
    inp = build_input_example(sample, evaluation_lib)
    prompt_to_response = {inp.prompt: generation}

    strict_out = evaluation_lib.test_instruction_following_strict(
        inp, prompt_to_response
    )
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
        "score": prompt_level_loose_acc,
        "is_pass": bool(prompt_level_loose_acc),
        "parsed": parsed,
    }


def aggregate_instruction_following(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Prompt level: mean over samples of each sample's mean over generations.

    Instruction level: micro average over instruction instances, each first
    averaged over the sample's generations.
    """
    del metric_options
    # This order fixes the output key order.
    values: dict[str, list[float]] = {
        "prompt_level_strict_acc": [],
        "inst_level_strict_acc": [],
        "prompt_level_loose_acc": [],
        "inst_level_loose_acc": [],
    }
    for item in sample_results:
        parsed = [record.parsed for record in to_records(item["records"])]
        if not parsed:
            continue
        for mode in ("strict", "loose"):
            values[f"prompt_level_{mode}_acc"].append(
                mean([float(p[f"prompt_level_{mode}_acc"]) for p in parsed])
            )
            inst_flags = [p[f"inst_level_{mode}_acc"] for p in parsed]
            for flags in zip(*inst_flags, strict=True):
                values[f"inst_level_{mode}_acc"].append(
                    mean([float(bool(flag)) for flag in flags])
                )

    result: dict[str, float] = {}
    for key, key_values in values.items():
        result[key] = mean(key_values)
        result[f"{key}_stderr"] = mean_stderr(key_values)
    return result
