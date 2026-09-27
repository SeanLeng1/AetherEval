"""Prompt, Base/Plus execution and aggregation shared by HumanEval+ and MBPP+.

EvalPlus is a scoring-only install and task.py imports build_prompt from here,
so the checker functions import it themselves.
"""

import json
from functools import lru_cache
from typing import Any, Callable

from aethereval.core.types import GenerationRecord, Sample
from aethereval.metrics.common import aggregate_binary_results, mean, mean_stderr, to_records


SCORING_PROTOCOL = "evalplus-26d6d00"


def build_prompt(sample: Sample) -> list[dict[str, str]]:
    # Short reasoning plus fenced code; no assistant/code prefill.
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


def _deserialize(dataset: str, task_id: str, inputs: Any) -> Any:
    from evalplus.data.mbpp import mbpp_deserialize_inputs

    return mbpp_deserialize_inputs(task_id, inputs) if dataset == "mbpp" else inputs


@lru_cache(maxsize=1024)
def _oracle(dataset: str, task_id: str, reference: str, entry_point: str,
            inputs_json: str, output_not_none: bool):
    from evalplus.gen.util import trusted_exec

    # Key by contents, not only task ID: a different dataset must not reuse stale outputs.
    # EvalPlus trusted_exec deep-copies each input before calling the reference.
    return trusted_exec(
        reference, _deserialize(dataset, task_id, json.loads(inputs_json)), entry_point,
        record_time=True, output_not_none=output_not_none,
    )


def score_base_plus(
    dataset: str,
    sample: Sample,
    solution: str,
    reference: str,
    output_not_none: bool = False,
) -> dict[str, Any]:
    """Run the official checker on Base, then on Plus only if Base passes."""
    from evalplus.eval import PASS, untrusted_check

    data = sample.data
    entry_point = str(data["entry_point"])
    statuses = {}
    for split in ("base", "plus"):
        if split == "plus" and statuses["base"] != PASS:
            statuses["plus"] = "skipped"
            break
        inputs = data[f"{split}_input"]
        expected, ref_time = _oracle(
            dataset, sample.id, reference, entry_point, json.dumps(inputs), output_not_none
        )
        statuses[split], _ = untrusted_check(
            dataset, solution, _deserialize(dataset, sample.id, inputs), entry_point,
            expected=expected, atol=float(data["atol"]), ref_time=ref_time, fast_check=True,
        )
    base_pass = statuses["base"] == PASS
    plus_pass = base_pass and statuses["plus"] == PASS
    return {
        "base_status": statuses["base"], "plus_status": statuses["plus"],
        "base_pass": base_pass, "plus_pass": plus_pass,
    }


def aggregate_base_plus(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None,
    parsed_flag_fn: Callable[[GenerationRecord], bool],
) -> dict[str, float]:
    result = aggregate_binary_results(
        sample_results, metric_options, parsed_flag_fn=parsed_flag_fn,
    )
    base_means = [
        mean([float((record.parsed or {}).get("base_pass", False))
              for record in to_records(item["records"])])
        for item in sample_results if item["records"]
    ]
    result.update(
        accuracy_plus=result["accuracy"], accuracy_plus_stderr=result["accuracy_stderr"],
        accuracy_base=mean(base_means), accuracy_base_stderr=mean_stderr(base_means),
    )
    return result
