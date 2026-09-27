from typing import Any

from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

from aethereval.core.types import Sample
from benchmark_utils.evalplus import SCORING_PROTOCOL, aggregate_base_plus, score_base_plus
from benchmark_utils.evalplus_sanitize import sanitize


PRIMARY_METRIC = "pass@1"
# EvalPlus sets a 4 GiB RLIMIT_AS in a checker started from the scoring process;
# score from a lean spawned worker even at num_proc=1 so the budget is the same.
SCORE_IN_SUBPROCESS = True


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    data = sample.data
    entry_point = data["entry_point"]
    solution = sanitize(generation, entrypoint=entry_point)
    parsed = score_base_plus(
        "mbpp", sample, solution, data["prompt"] + data["canonical_solution"],
        output_not_none=entry_point in MBPP_OUTPUT_NOT_NONE_TASKS,
    )
    parsed["had_code"] = bool(solution.strip())
    return {
        "score": float(parsed["plus_pass"]), "is_pass": parsed["plus_pass"], "parsed": parsed,
        "meta": {"scoring_protocol": SCORING_PROTOCOL, "source_version": "v0.2.0"},
    }


def aggregate(sample_results, metric_options=None) -> dict[str, float]:
    return aggregate_base_plus(
        sample_results, metric_options,
        parsed_flag_fn=lambda record: bool((record.parsed or {}).get("had_code")),
    )
