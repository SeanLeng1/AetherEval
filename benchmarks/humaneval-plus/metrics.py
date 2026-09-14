import json
import re
from functools import lru_cache
from typing import Any

from evalplus.eval import PASS, untrusted_check
from evalplus.gen.util import trusted_exec

from aethereval.metrics.common import (
    aggregate_binary_results,
    mean,
    mean_stderr,
    to_records,
)
from aethereval.core.types import GenerationRecord, Sample

PRIMARY_METRIC = "pass@1"
SCORING_PROTOCOL = "evalplus-0.3.1"

_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL
)
_ANSWER_BLOCK_RE = re.compile(
    r"here is the completed function:\s*```(?:python)?[ \t]*\n?(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


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


def _strip_fence_wrapper(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text.rstrip()

    lines = stripped.splitlines()
    if not lines:
        return ""

    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).rstrip()


def _extract_python_candidate(text: str) -> str:
    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        return answer_match.group(1).rstrip()

    blocks = [m.group(1).rstrip() for m in _CODE_BLOCK_RE.finditer(text)]
    if blocks:
        # Prefer the final block to match common "reasoning + final code block" outputs.
        return blocks[-1]

    lower = text.lower()
    marker = "here is the completed function:"
    marker_idx = lower.find(marker)
    if marker_idx >= 0:
        return text[marker_idx + len(marker) :].rstrip()

    return text.rstrip()


def _candidate_solution(sample: Sample, generation: str) -> tuple[str, bool]:
    prompt = str(sample.data["prompt"])
    entry_point = str(sample.data["entry_point"])

    extracted = _extract_python_candidate(generation)
    extracted = _strip_fence_wrapper(extracted)
    if not extracted.strip():
        return prompt, False

    continuation = extracted
    is_full_solution = False
    if continuation.startswith(prompt):
        continuation = continuation[len(prompt) :]
        is_full_solution = True
    elif prompt in continuation:
        continuation = continuation.split(prompt, 1)[1]
        is_full_solution = True
    elif re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", continuation):
        is_full_solution = True

    continuation = continuation.lstrip("\n")
    if not continuation:
        return prompt, is_full_solution

    joiner = "" if prompt.endswith("\n") else "\n"
    # Match OLMES HumanEval behavior: always execute prompt + continuation.
    return prompt + joiner + continuation, is_full_solution


@lru_cache(maxsize=256)
def _oracle(reference: str, entry_point: str, inputs_json: str):
    # Key by contents, not task ID: a different dataset must not reuse stale outputs.
    # EvalPlus trusted_exec deep-copies each input before calling the reference.
    return trusted_exec(reference, json.loads(inputs_json), entry_point, record_time=True)


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    data = sample.data
    entry_point = str(data["entry_point"])
    solution, is_full_solution = _candidate_solution(sample, generation)
    prompt = str(data["prompt"])
    reference = (
        prompt
        + ("" if prompt.endswith("\n") else "\n")
        + str(data["canonical_solution"])
    )
    statuses = {}
    for split in ("base", "plus"):
        if split == "plus" and statuses["base"] != PASS:
            statuses["plus"] = "skipped"
            break
        inputs = data[f"{split}_input"]
        if not inputs:
            raise ValueError(f"{sample.id}: {split}_input is empty")
        expected, ref_time = _oracle(reference, entry_point, json.dumps(inputs))
        statuses[split], _ = untrusted_check(
            "humaneval",
            solution,
            inputs,
            entry_point,
            expected=expected,
            atol=float(data["atol"]),
            ref_time=ref_time,
            fast_check=True,
        )

    base_pass = statuses["base"] == PASS
    plus_pass = base_pass and statuses["plus"] == PASS
    parsed = {
        "base_status": statuses["base"],
        "plus_status": statuses["plus"],
        "base_pass": base_pass,
        "plus_pass": plus_pass,
    }
    return {
        "score": float(plus_pass),
        "is_pass": plus_pass,
        "parsed": parsed,
        "meta": {
            **parsed,
            "scoring_protocol": SCORING_PROTOCOL,
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
        records = to_records(item["records"])
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
