import math
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ..core.types import GenerationRecord


def to_records(raw_records: list[dict[str, Any]]) -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for rec in raw_records:
        meta = rec["meta"]
        if not isinstance(meta, dict):
            raise ValueError("Generation record meta must be a dict")
        records.append(
            GenerationRecord(
                sample_id=str(rec["sample_id"]),
                gen_idx=int(rec["gen_idx"]),
                prompt=rec["prompt"],
                generation=rec["generation"],
                score=float(rec["score"]),
                is_pass=bool(rec["is_pass"]),
                parsed=rec.get("parsed"),
                gold=rec.get("gold"),
                error=rec.get("error"),
                meta=meta,
            )
        )
    records.sort(key=lambda x: x.gen_idx)
    return records


_THINK_END = "</think>"


def strip_reasoning(text: str) -> str:
    """Keep text after the last closing tag; without one, preserve the response.

    No opening tag is required. This is a text-level convention, so even a
    literal closing tag in generated code is treated as an answer boundary.
    """
    _, marker, answer = text.rpartition(_THINK_END)
    return answer.lstrip() if marker else text


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


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


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
        records = to_records(item["records"])
        if not records:
            continue

        record_scores = [float(value_fn(record)) for record in records]
        record_parsed_flags = [1.0 if parsed_fn(record) else 0.0 for record in records]

        sample_acc = mean(record_scores)
        sample_parsed = mean(record_parsed_flags)
        sample_acc_values.append(sample_acc)
        sample_parsed_values.append(sample_parsed)
        sample_binary_scores.append(
            [1 if score >= 1.0 else 0 for score in record_scores]
        )

        if group_key:
            sample_meta = item["meta"]
            if not isinstance(sample_meta, dict):
                raise ValueError("sample_results meta must be a dict")
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
        pass_metrics[k] = [
            pass_at_k(binary_scores, k) for binary_scores in sample_binary_scores
        ]

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
