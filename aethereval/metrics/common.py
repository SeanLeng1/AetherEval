from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ..core.types import GenerationRecord, Sample


def to_records(raw_records: list[dict[str, Any]]) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for rec in raw_records:
        records.append(
            GenerationRecord(
                sample_id=str(rec["sample_id"]),
                gen_idx=int(rec["gen_idx"]),
                prompt=rec.get("prompt", ""),
                generation=rec.get("generation", ""),
                score=float(rec.get("score", 0.0)),
                is_pass=bool(rec.get("is_pass", False)),
                parsed=rec.get("parsed"),
                gold=rec.get("gold"),
                error=rec.get("error"),
                meta=rec.get("meta", {}) if isinstance(rec.get("meta", {}), dict) else {},
            )
        )
    records.sort(key=lambda x: x.gen_idx)
    return records


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def mean_stderr(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean(values)
    variance = math.fsum((x - mu) ** 2 for x in values) / (n - 1)
    return math.sqrt(max(variance, 0.0)) / math.sqrt(n)


def pass_at_k(binary_scores: list[int], k: int) -> float:
    n = len(binary_scores)
    if n == 0:
        return 0.0
    c = binary_scores.count(1)
    if n - c < k:
        return 1.0

    product = 1.0
    for i in range(n - c + 1, n + 1):
        product *= 1.0 - (k / float(i))
    return 1.0 - product


def default_pass_k_values(n: int) -> list[int]:
    if n <= 0:
        return []
    values: list[int] = []
    k = 1
    while k <= n:
        values.append(k)
        k *= 2
    if values[-1] != n:
        values.append(n)
    return values


def resolve_pass_k_values(raw: Any, n: int) -> list[int]:
    if raw is None:
        values = default_pass_k_values(n)
    elif isinstance(raw, str):
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [int(x) for x in raw]
    else:
        raise ValueError("pass_k_values must be list[int] or comma-separated string")

    cleaned = sorted({k for k in values if k >= 1})
    return [k for k in cleaned if k <= n]


def _normalize_instruction_flags(
    raw_values: Any,
    expected_count: int | None,
    fallback: float,
) -> list[float]:
    values: list[float] = []
    if isinstance(raw_values, list):
        values = [float(bool(x)) for x in raw_values]

    if expected_count is not None and expected_count > 0:
        if not values:
            return [fallback for _ in range(expected_count)]
        if len(values) < expected_count:
            return values + [fallback for _ in range(expected_count - len(values))]
        if len(values) > expected_count:
            return values[:expected_count]
        return values

    if values:
        return values
    return [fallback]


def aggregate_instruction_following_results(
    sample_results: list[dict[str, Any]],
) -> dict[str, float]:
    if not sample_results:
        return {
            "prompt_level_strict_acc": 0.0,
            "prompt_level_strict_acc_stderr": 0.0,
            "inst_level_strict_acc": 0.0,
            "inst_level_strict_acc_stderr": 0.0,
            "prompt_level_loose_acc": 0.0,
            "prompt_level_loose_acc_stderr": 0.0,
            "inst_level_loose_acc": 0.0,
            "inst_level_loose_acc_stderr": 0.0,
        }

    prompt_level_strict_values: list[float] = []
    prompt_level_loose_values: list[float] = []
    # These values are instruction-level (micro) aggregates.
    inst_level_strict_values: list[float] = []
    inst_level_loose_values: list[float] = []

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        sample_meta = item.get("meta", {}) if isinstance(item.get("meta", {}), dict) else {}
        raw_instruction_ids = sample_meta.get("instruction_id_list")
        expected_instruction_count = len(raw_instruction_ids) if isinstance(raw_instruction_ids, list) else None
        if expected_instruction_count == 0:
            expected_instruction_count = None

        sample_prompt_strict_values: list[float] = []
        sample_prompt_loose_values: list[float] = []
        sample_inst_strict_lists: list[list[float]] = []
        sample_inst_loose_lists: list[list[float]] = []

        for record in records:
            parsed = record.parsed if isinstance(record.parsed, dict) else {}

            prompt_strict = float(parsed.get("prompt_level_strict_acc", record.score))
            prompt_loose = float(parsed.get("prompt_level_loose_acc", prompt_strict))
            sample_prompt_strict_values.append(prompt_strict)
            sample_prompt_loose_values.append(prompt_loose)

            sample_inst_strict_lists.append(
                _normalize_instruction_flags(
                    parsed.get("inst_level_strict_acc"),
                    expected_instruction_count,
                    prompt_strict,
                )
            )
            sample_inst_loose_lists.append(
                _normalize_instruction_flags(
                    parsed.get("inst_level_loose_acc"),
                    expected_instruction_count,
                    prompt_loose,
                )
            )

        prompt_level_strict_values.append(mean(sample_prompt_strict_values))
        prompt_level_loose_values.append(mean(sample_prompt_loose_values))

        max_inst_count = max((len(values) for values in sample_inst_strict_lists), default=0)
        for inst_idx in range(max_inst_count):
            strict_across_records: list[float] = []
            loose_across_records: list[float] = []
            for rec_idx in range(len(sample_inst_strict_lists)):
                strict_values = sample_inst_strict_lists[rec_idx]
                loose_values = sample_inst_loose_lists[rec_idx]

                strict_across_records.append(
                    strict_values[inst_idx]
                    if inst_idx < len(strict_values)
                    else sample_prompt_strict_values[rec_idx]
                )
                loose_across_records.append(
                    loose_values[inst_idx]
                    if inst_idx < len(loose_values)
                    else sample_prompt_loose_values[rec_idx]
                )

            inst_level_strict_values.append(mean(strict_across_records))
            inst_level_loose_values.append(mean(loose_across_records))

    return {
        "prompt_level_strict_acc": mean(prompt_level_strict_values),
        "prompt_level_strict_acc_stderr": mean_stderr(prompt_level_strict_values),
        "inst_level_strict_acc": mean(inst_level_strict_values),
        "inst_level_strict_acc_stderr": mean_stderr(inst_level_strict_values),
        "prompt_level_loose_acc": mean(prompt_level_loose_values),
        "prompt_level_loose_acc_stderr": mean_stderr(prompt_level_loose_values),
        "inst_level_loose_acc": mean(inst_level_loose_values),
        "inst_level_loose_acc_stderr": mean_stderr(inst_level_loose_values),
    }


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalize_choice(value: str, valid_set: set[str]) -> str | None:
    text = value.strip().upper()
    text = text.replace("(", "").replace(")", "")
    text = text.replace(".", "").replace(":", "")
    if not text:
        return None
    head = text[0]
    return head if head in valid_set else None


@lru_cache(maxsize=32)
def _choice_patterns(valid_letters: str) -> list[tuple[str, re.Pattern[str]]]:
    char_class = "".join(re.escape(letter) for letter in valid_letters)
    # Match standalone choice letters only (avoid picking letters inside words like "because").
    choice_re = rf"(?<![A-Za-z0-9])\(?[{char_class}]\)?(?![A-Za-z0-9])"
    return [
        (
            "final_answer",
            re.compile(
                rf"(?i)final\s+answer(?:\s+is)?\s*[:：]?\s*(?P<choice>{choice_re})"
            ),
        ),
        (
            "answer_colon",
            re.compile(rf"(?i)\banswer\s*[:：]\s*(?P<choice>{choice_re})"),
        ),
        (
            "answer_anchor",
            re.compile(rf"(?i)\banswer\b.{{0,80}}?(?P<choice>{choice_re})"),
        ),
        (
            "option_anchor",
            re.compile(
                rf"(?i)\b(?:option|choice)\b(?:\s+is)?\s*(?P<choice>{choice_re})\b"
            ),
        ),
        (
            "select_anchor",
            re.compile(
                rf"(?i)\b(?:choose|chosen|select|selected|pick|picked)\b.{{0,40}}?(?P<choice>{choice_re})"
            ),
        ),
        (
            "line_start",
            re.compile(
                rf"(?im)^\s*(?:\*\*)?\s*(?P<choice>{choice_re})(?:\*\*)?\s*(?:[\)\].,:]|$)"
            ),
        ),
    ]


def extract_choice(
    text: str,
    choices: dict[str, str],
    valid_letters: list[str],
) -> tuple[str | None, str]:
    del choices
    valid_set = set(valid_letters)
    patterns = _choice_patterns("".join(valid_letters))

    candidates: list[tuple[int, int, str, str]] = []
    for priority, (method, pattern) in enumerate(patterns):
        for match in pattern.finditer(text):
            choice = _normalize_choice(match.group("choice"), valid_set)
            if choice is None:
                continue
            candidates.append((priority, -match.end(), choice, method))

    if candidates:
        candidates.sort()
        _, _, choice, method = candidates[0]
        return choice, method

    return None, "none"


def score_generation_mcq(sample: Sample, generation: str) -> dict[str, Any]:
    choices = sample.data.get("choices", {})
    if not isinstance(choices, dict) or not choices:
        choices = {}

    valid_letters = [k.strip().upper() for k in sorted(choices.keys()) if str(k).strip()]
    if not valid_letters:
        valid_letters = ["A", "B", "C", "D"]

    prediction, method = extract_choice(generation, choices, valid_letters)
    gold = str(sample.gold).strip().upper()
    score = 1.0 if prediction == gold else 0.0

    parsed = {
        "prediction": prediction,
        "gold": gold,
        "extract_method": method,
    }
    return {
        "score": score,
        "is_pass": bool(score),
        "parsed": parsed,
        "meta": {
            "prediction": prediction,
            "extract_method": method,
        },
    }


def _default_mcq_parsed_flag(record: GenerationRecord) -> bool:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    prediction = parsed.get("prediction")
    return isinstance(prediction, str) and bool(prediction)


def aggregate_binary_results(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
    *,
    score_fn: Callable[[GenerationRecord], float] | None = None,
    parsed_flag_fn: Callable[[GenerationRecord], bool] | None = None,
    group_key: str | None = None,
    group_metric_prefix: str = "accuracy_",
) -> dict[str, float]:
    options = metric_options or {}
    n_hint = int(options.get("n", 0)) if options.get("n") is not None else 0

    if not sample_results:
        return {
            "accuracy": 0.0,
            "accuracy_stderr": 0.0,
            "parsed_rate": 0.0,
            "pass@1": 0.0,
            "pass@1_stderr": 0.0,
        }

    sample_acc_values: list[float] = []
    sample_parsed_values: list[float] = []
    sample_binary_scores: list[list[int]] = []
    grouped_scores: dict[str, list[float]] = defaultdict(list)

    value_fn = score_fn or (lambda record: float(record.score))
    parsed_fn = parsed_flag_fn or _default_mcq_parsed_flag

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        record_scores = [float(value_fn(record)) for record in records]
        record_parsed_flags = [1.0 if parsed_fn(record) else 0.0 for record in records]

        sample_acc = mean(record_scores)
        sample_parsed = mean(record_parsed_flags)
        sample_acc_values.append(sample_acc)
        sample_parsed_values.append(sample_parsed)
        sample_binary_scores.append([1 if score >= 1.0 else 0 for score in record_scores])

        if group_key:
            sample_meta = item.get("meta", {}) if isinstance(item.get("meta", {}), dict) else {}
            raw_group = str(sample_meta.get(group_key, "")).strip()
            group_name = _slugify(raw_group)
            if group_name:
                grouped_scores[group_name].append(sample_acc)

    if not sample_acc_values:
        return {
            "accuracy": 0.0,
            "accuracy_stderr": 0.0,
            "parsed_rate": 0.0,
            "pass@1": 0.0,
            "pass@1_stderr": 0.0,
        }

    n_ref = n_hint if n_hint > 0 else max(len(x) for x in sample_binary_scores)
    pass_k_values = resolve_pass_k_values(options.get("pass_k_values"), n_ref)
    if 1 not in pass_k_values and n_ref >= 1:
        pass_k_values = [1] + pass_k_values

    pass_metrics: dict[int, list[float]] = defaultdict(list)
    for k in pass_k_values:
        # Match official EvalPlus/LiveCodeBench convention:
        # only report pass@k when every sample has at least k generations.
        if not all(len(binary_scores) >= k for binary_scores in sample_binary_scores):
            continue
        pass_metrics[k] = [pass_at_k(binary_scores, k) for binary_scores in sample_binary_scores]

    result: dict[str, float] = {
        "accuracy": mean(sample_acc_values),
        "accuracy_stderr": mean_stderr(sample_acc_values),
        "parsed_rate": mean(sample_parsed_values),
    }
    if n_ref > 1:
        result[f"accuracy@{n_ref}"] = result["accuracy"]

    for k in sorted(pass_metrics):
        values = pass_metrics[k]
        result[f"pass@{k}"] = mean(values)
        result[f"pass@{k}_stderr"] = mean_stderr(values)

    for group_name in sorted(grouped_scores):
        result[f"{group_metric_prefix}{group_name}"] = mean(grouped_scores[group_name])

    return result


def aggregate_mcq_results(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
    *,
    group_key: str | None = None,
    group_metric_prefix: str = "accuracy_",
) -> dict[str, float]:
    return aggregate_binary_results(
        sample_results,
        metric_options,
        parsed_flag_fn=_default_mcq_parsed_flag,
        group_key=group_key,
        group_metric_prefix=group_metric_prefix,
    )
