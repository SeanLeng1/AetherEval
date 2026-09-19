import json
from functools import lru_cache
from typing import Any

from evalplus.data.mbpp import mbpp_deserialize_inputs
from evalplus.eval import PASS, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec
from evalplus.sanitize import sanitize

from aethereval.core.types import Sample
from aethereval.metrics.common import aggregate_binary_results, mean, mean_stderr, to_records


PRIMARY_METRIC = "pass@1"


@lru_cache(maxsize=1024)
def _oracle(task_id: str, reference: str, entry_point: str, inputs_json: str):
    return trusted_exec(
        reference,
        mbpp_deserialize_inputs(task_id, json.loads(inputs_json)),
        entry_point,
        record_time=True,
        output_not_none=entry_point in MBPP_OUTPUT_NOT_NONE_TASKS,
    )


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    data = sample.data
    entry_point = data["entry_point"]
    solution = sanitize(generation, entrypoint=entry_point)
    reference = data["prompt"] + data["canonical_solution"]
    statuses = {}
    for split in ("base", "plus"):
        if split == "plus" and statuses["base"] != PASS:
            statuses["plus"] = "skipped"
            break
        raw_inputs = data[f"{split}_input"]
        expected, ref_time = _oracle(
            sample.id, reference, entry_point, json.dumps(raw_inputs)
        )
        statuses[split], _ = untrusted_check(
            "mbpp", solution,
            mbpp_deserialize_inputs(sample.id, raw_inputs),
            entry_point, expected=expected, atol=float(data["atol"]),
            ref_time=ref_time, fast_check=True,
        )
    base_pass = statuses["base"] == PASS
    plus_pass = base_pass and statuses["plus"] == PASS
    parsed = {
        "base_status": statuses["base"], "plus_status": statuses["plus"],
        "base_pass": base_pass, "plus_pass": plus_pass,
        "had_code": bool(solution.strip()),
    }
    return {
        "score": float(plus_pass), "is_pass": plus_pass, "parsed": parsed,
        "meta": {"scoring_protocol": "evalplus-26d6d00", "source_version": "v0.2.0"},
    }


def aggregate(sample_results, metric_options=None) -> dict[str, float]:
    result = aggregate_binary_results(
        sample_results, metric_options,
        parsed_flag_fn=lambda record: bool((record.parsed or {}).get("had_code")),
    )
    base_means = [
        mean([float((record.parsed or {}).get("base_pass", False))
              for record in to_records(item["records"])])
        for item in sample_results if item["records"]
    ]
    result.update(
        accuracy_base=mean(base_means), accuracy_base_stderr=mean_stderr(base_means),
        accuracy_plus=result["accuracy"], accuracy_plus_stderr=result["accuracy_stderr"],
    )
    return result
