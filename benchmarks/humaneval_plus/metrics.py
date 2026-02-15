from __future__ import annotations

import re
from typing import Any

from aethereval.metrics.common import aggregate_binary_results, mean, mean_stderr, to_records
from aethereval.core.types import GenerationRecord, Sample

try:
    from evalplus.config import DEFAULT_GT_TIME_LIMIT_FACTOR, DEFAULT_MIN_TIME_LIMIT
    from evalplus.eval import PASS, untrusted_check as _evalplus_untrusted_check
    from evalplus.gen.util import trusted_exec
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "evalplus is required for humaneval_plus metrics. Install with `pip install evalplus`."
    ) from exc


PRIMARY_METRIC = "pass@1"


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ORACLE_CACHE: dict[str, dict[str, Any]] = {}


def _empty_aggregate_result() -> dict[str, float]:
    return {
        "accuracy": 0.0,
        "accuracy_stderr": 0.0,
        "accuracy_plus": 0.0,
        "accuracy_plus_stderr": 0.0,
        "accuracy_base": 0.0,
        "accuracy_base_stderr": 0.0,
        "pass@1": 0.0,
        "pass@1_stderr": 0.0,
    }


def _untrusted_check(
    code: str,
    inputs: list[Any],
    entry_point: str,
    *,
    expected: list[Any],
    atol: float,
    ref_time: list[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> tuple[str, list[bool]]:
    status, details = _evalplus_untrusted_check(
        "humaneval",
        code,
        inputs,
        entry_point,
        expected=expected,
        atol=atol,
        ref_time=ref_time,
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )
    return status, [bool(x) for x in list(details)]


def _record_plus_score(record: GenerationRecord) -> float:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    plus_pass = bool(parsed.get("plus_pass", bool(record.score >= 1.0)))
    return 1.0 if plus_pass else 0.0


def _record_base_score(record: GenerationRecord) -> float:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    plus_pass = bool(parsed.get("plus_pass", bool(record.score >= 1.0)))
    base_pass = bool(parsed.get("base_pass", plus_pass))
    return 1.0 if base_pass else 0.0


def _record_has_parsed(record: GenerationRecord) -> bool:
    return isinstance(record.parsed, dict) and bool(record.parsed)


def _normalize_inputs(raw_inputs: list[Any]) -> list[list[Any]]:
    normalized: list[list[Any]] = []
    for item in raw_inputs:
        if isinstance(item, (list, tuple)):
            normalized.append(list(item))
        else:
            normalized.append([item])
    return normalized


def _extract_python_block(text: str) -> str:
    matches = [m.group(1).rstrip() for m in _CODE_BLOCK_RE.finditer(text)]
    if matches:
        return max(matches, key=len)
    return text.rstrip()


def _candidate_solution(sample: Sample, generation: str) -> tuple[str, bool]:
    prompt = str(sample.data["prompt"])
    entry_point = str(sample.data["entry_point"])

    extracted = _extract_python_block(generation)
    if not extracted.strip():
        return prompt, False

    if prompt in extracted:
        return extracted, True

    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", extracted):
        return extracted, True

    # Treat as completion continuation.
    joiner = "" if prompt.endswith("\n") else "\n"
    return prompt + joiner + extracted, False


def _oracle(sample: Sample) -> dict[str, Any]:
    sample_id = sample.id
    if sample_id in _ORACLE_CACHE:
        return _ORACLE_CACHE[sample_id]

    prompt = str(sample.data["prompt"])
    canonical_solution = str(sample.data["canonical_solution"])
    entry_point = str(sample.data["entry_point"])
    base_input = _normalize_inputs(list(sample.data["base_input"]))
    plus_input = _normalize_inputs(list(sample.data["plus_input"]))

    reference_code = prompt + ("" if prompt.endswith("\n") else "\n") + canonical_solution
    base_expected, base_time = trusted_exec(
        reference_code,
        base_input,
        entry_point,
        record_time=True,
    )
    plus_expected, plus_time = trusted_exec(
        reference_code,
        plus_input,
        entry_point,
        record_time=True,
    )

    out = {
        "base_input": base_input,
        "plus_input": plus_input,
        "base_expected": base_expected,
        "plus_expected": plus_expected,
        "base_time": base_time,
        "plus_time": plus_time,
    }
    _ORACLE_CACHE[sample_id] = out
    return out


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    entry_point = str(sample.data["entry_point"])
    atol = float(sample.data.get("atol", 0.0))

    oracle = _oracle(sample)
    solution, is_full_solution = _candidate_solution(sample, generation)

    base_status, _ = _untrusted_check(
        solution,
        oracle["base_input"],
        entry_point,
        expected=oracle["base_expected"],
        atol=atol,
        ref_time=oracle["base_time"],
        fast_check=True,
    )

    if base_status != PASS:
        plus_status = base_status
    else:
        plus_status, _ = _untrusted_check(
            solution,
            oracle["plus_input"],
            entry_point,
            expected=oracle["plus_expected"],
            atol=atol,
            ref_time=oracle["plus_time"],
            fast_check=True,
        )

    base_pass = base_status == PASS
    plus_pass = base_pass and (plus_status == PASS)

    parsed = {
        "base_status": base_status,
        "plus_status": plus_status,
        "base_pass": base_pass,
        "plus_pass": plus_pass,
    }
    return {
        "score": 1.0 if plus_pass else 0.0,
        "is_pass": plus_pass,
        "parsed": parsed,
        "meta": {
            "base_pass": base_pass,
            "plus_pass": plus_pass,
            "full_solution": is_full_solution,
        },
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    if not sample_results:
        return _empty_aggregate_result()

    base_means: list[float] = []
    plus_metrics = aggregate_binary_results(
        sample_results,
        metric_options,
        score_fn=_record_plus_score,
        parsed_flag_fn=_record_has_parsed,
    )

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        base_scores = [_record_base_score(record) for record in records]
        base_means.append(mean(base_scores))

    if not base_means:
        return _empty_aggregate_result()

    accuracy_plus = float(plus_metrics.get("accuracy", 0.0))
    accuracy_plus_stderr = float(plus_metrics.get("accuracy_stderr", 0.0))
    result: dict[str, float] = dict(plus_metrics)
    result.update(
        {
            "accuracy": accuracy_plus,
            "accuracy_stderr": accuracy_plus_stderr,
            "accuracy_plus": accuracy_plus,
            "accuracy_plus_stderr": accuracy_plus_stderr,
            "accuracy_base": mean(base_means),
            "accuracy_base_stderr": mean_stderr(base_means),
        }
    )

    return result
