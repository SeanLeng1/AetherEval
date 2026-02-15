from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aethereval.metrics.common import mean, mean_stderr, pass_at_k, resolve_pass_k_values, to_records
from aethereval.core.types import Sample


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from eval_runtime import PASS, trusted_exec, untrusted_check


PRIMARY_METRIC = "pass@1"


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ORACLE_CACHE: dict[str, dict[str, Any]] = {}


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

    base_status, _ = untrusted_check(
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
        plus_status, _ = untrusted_check(
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
    options = metric_options or {}
    n_hint = int(options.get("n", 0)) if options.get("n") is not None else 0

    if not sample_results:
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

    plus_means: list[float] = []
    base_means: list[float] = []
    parsed_flags: list[float] = []
    sample_binary_plus: list[list[int]] = []

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        plus_scores: list[float] = []
        base_scores: list[float] = []
        parsed_local: list[float] = []

        for record in records:
            parsed = record.parsed if isinstance(record.parsed, dict) else {}
            plus_pass = bool(parsed.get("plus_pass", bool(record.score >= 1.0)))
            base_pass = bool(parsed.get("base_pass", plus_pass))
            plus_scores.append(1.0 if plus_pass else 0.0)
            base_scores.append(1.0 if base_pass else 0.0)
            parsed_local.append(1.0 if isinstance(parsed, dict) and parsed else 0.0)

        plus_means.append(mean(plus_scores))
        base_means.append(mean(base_scores))
        parsed_flags.append(mean(parsed_local))
        sample_binary_plus.append([int(x >= 1.0) for x in plus_scores])

    if not plus_means:
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

    n_ref = n_hint if n_hint > 0 else max(len(x) for x in sample_binary_plus)
    pass_k_values = resolve_pass_k_values(options.get("pass_k_values"), n_ref)
    if 1 not in pass_k_values and n_ref >= 1:
        pass_k_values = [1] + pass_k_values

    pass_metrics: dict[int, list[float]] = defaultdict(list)
    for binary_scores in sample_binary_plus:
        for k in pass_k_values:
            if k <= len(binary_scores):
                pass_metrics[k].append(pass_at_k(binary_scores, k))

    accuracy_plus = mean(plus_means)
    accuracy_plus_stderr = mean_stderr(plus_means)
    result: dict[str, float] = {
        "accuracy": accuracy_plus,
        "accuracy_stderr": accuracy_plus_stderr,
        "accuracy_plus": accuracy_plus,
        "accuracy_plus_stderr": accuracy_plus_stderr,
        "accuracy_base": mean(base_means),
        "accuracy_base_stderr": mean_stderr(base_means),
        "parsed_rate": mean(parsed_flags),
    }

    if n_ref > 1:
        result[f"accuracy@{n_ref}"] = accuracy_plus

    for k in sorted(pass_metrics):
        values = pass_metrics[k]
        result[f"pass@{k}"] = mean(values)
        result[f"pass@{k}_stderr"] = mean_stderr(values)

    return result
