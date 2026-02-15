from __future__ import annotations

from typing import Any

from aethereval.metrics.common import aggregate_binary_results
from aethereval.core.types import GenerationRecord, Sample
from benchmarks.livecodebench.lcb_eval_runtime import evaluate_candidate


PRIMARY_METRIC = "pass@1"


def _extract_code(text: str) -> tuple[str, str]:
    output_lines = text.split("\n")
    fence_lines = [i for i, line in enumerate(output_lines) if "```" in line]
    if len(fence_lines) < 2:
        return "", "no_fenced_code"

    start, end = fence_lines[-2], fence_lines[-1]
    if end <= start:
        return "", "invalid_fenced_code"

    return "\n".join(output_lines[start + 1 : end]).rstrip(), "fenced_last_block"


def _is_pass_status(value: int | bool) -> bool:
    if isinstance(value, bool):
        return value
    return int(value) > 0


def _first_error_code(statuses: list[int | bool]) -> int | None:
    for value in statuses:
        if isinstance(value, bool):
            continue
        return int(value)
    return None


def _parsed_has_code(record: GenerationRecord) -> bool:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    return bool(parsed.get("had_code", False))


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    code, extract_method = _extract_code(generation)
    had_code = bool(code.strip())
    if not had_code:
        return {
            "score": 0.0,
            "is_pass": False,
            "parsed": {
                "extract_method": extract_method,
                "had_code": False,
                "num_tests": len(sample.data.get("outputs", [])),
                "passed_tests": 0,
                "statuses": [],
            },
            "meta": {"error": "empty_generation"},
        }

    timeout = int(sample.data.get("timeout_sec", 6))
    inputs = list(sample.data.get("inputs", []))
    outputs = list(sample.data.get("outputs", []))
    fn_name = sample.data.get("fn_name")
    fn_name = str(fn_name) if fn_name is not None else None

    statuses, runtime_error = evaluate_candidate(
        code=code,
        inputs=inputs,
        outputs=outputs,
        fn_name=fn_name,
        timeout=timeout,
    )
    passed = bool(statuses) and all(_is_pass_status(value) for value in statuses)
    passed_tests = sum(1 for value in statuses if _is_pass_status(value))

    parsed = {
        "extract_method": extract_method,
        "had_code": True,
        "num_tests": len(statuses),
        "passed_tests": passed_tests,
        "statuses": statuses,
    }
    meta: dict[str, Any] = {
        "error_code": _first_error_code(statuses),
    }
    if runtime_error:
        meta["runtime_error"] = runtime_error

    return {
        "score": 1.0 if passed else 0.0,
        "is_pass": passed,
        "parsed": parsed,
        "meta": meta,
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_binary_results(
        sample_results,
        metric_options,
        parsed_flag_fn=_parsed_has_code,
        group_key="platform",
    )
