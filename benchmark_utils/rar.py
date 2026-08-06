"""Shared task and LLM-judge implementation for ScaleAI RaR benchmarks."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import GenerationOutput, Sample
from benchmark_utils.llm_judge import (
    NORMAL_FORMAT_ATTEMPTS,
    chat_completion,
    local_constraint_body,
    parallel_map,
    resolve_judge_settings,
)
from benchmark_utils.rar_data import FAMILY_NAMES, FAMILY_POINTS, parse_domain
from benchmark_utils.rar_protocol import build_grader_prompt


_GEMMA_THOUGHT_CLOSE = "<channel|>"
_REASONING_BLOCK_RE = re.compile(
    rf"<think>.*?</think>|<\|channel>thought\n?.*?"
    rf"{re.escape(_GEMMA_THOUGHT_CLOSE)}",
    re.DOTALL,
)


def load_rar_samples(
    task_dir: Path, data_file: str, *, expected_domain: str
) -> list[Sample]:
    expected_domain = parse_domain(expected_domain)
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / data_file):
        if not isinstance(row, dict):
            raise ValueError("RaR rows must be JSON objects")
        sample_id = str(row["id"])
        prompt = _messages(row["prompt"], sample_id)
        rubrics = _validate_rubrics(row["rubrics"], sample_id)
        meta = row.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"RaR sample {sample_id} meta must be an object")
        domain = parse_domain(str(meta.get("domain", expected_domain)))
        if domain != expected_domain:
            raise ValueError(
                f"RaR sample {sample_id} has domain {domain}, expected "
                f"{expected_domain}"
            )
        samples.append(
            Sample(
                id=sample_id,
                gold=str(row["reference_answer"]),
                data={"prompt": prompt, "rubrics": rubrics},
                meta={**meta, "domain": domain, "rubric_count": len(rubrics)},
            )
        )
    return samples


def build_rar_prompt(sample: Sample) -> list[dict[str, str]]:
    return _messages(sample.data["prompt"], sample.id)


def _messages(raw: Any, sample_id: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"RaR prompt must be a nonempty message list: {sample_id}")
    result: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, dict):
            raise ValueError(f"RaR prompt message must be an object: {sample_id}")
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            raise ValueError(f"RaR prompt message is incomplete: {sample_id}")
        result.append({"role": role, "content": content})
    return result


def _validate_rubrics(raw: Any, sample_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"RaR sample {sample_id} must have nonempty rubrics")
    result: list[dict[str, Any]] = []
    for rubric in raw:
        if not isinstance(rubric, dict):
            raise ValueError(f"RaR rubric must be an object: {sample_id}")
        criterion = str(rubric.get("criterion", "")).strip()
        family = str(rubric.get("family", "")).strip()
        if not criterion or family not in FAMILY_POINTS:
            raise ValueError(f"RaR sample {sample_id} has an invalid rubric")
        points = float(rubric["points"])
        if not _close(points, FAMILY_POINTS[family]):
            raise ValueError(
                f"RaR sample {sample_id} rubric points {points} disagree with "
                f"family {family}"
            )
        result.append(
            {
                "criterion": criterion,
                "points": points,
                "family": family,
                "raw_weight": int(rubric["raw_weight"]),
                "title": str(rubric.get("title", "")),
            }
        )
    return result


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def parse_presence_response(text: str, expected_count: int) -> list[bool]:
    if not isinstance(text, str):
        raise TypeError("RaR grader content must be text")
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    close = cleaned.rfind(_GEMMA_THOUGHT_CLOSE)
    if close >= 0:
        cleaned = cleaned[close + len(_GEMMA_THOUGHT_CLOSE) :]
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE
    )
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
    if candidate is None:
        raise ValueError("RaR grader response contains no JSON object")
    try:
        payload = json.loads(re.sub(r",\s*}", "}", candidate))
    except json.JSONDecodeError as exc:
        raise ValueError("RaR grader returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("RaR grader JSON must be an object")
    expected = {str(index) for index in range(1, expected_count + 1)}
    if set(payload) != expected:
        raise ValueError(
            f"RaR grader returned keys {sorted(payload)}, expected {sorted(expected)}"
        )
    judgments: list[bool] = []
    for index in range(1, expected_count + 1):
        raw_value = payload[str(index)]
        if isinstance(raw_value, bool):
            judgments.append(raw_value)
            continue
        value = str(raw_value).strip().upper().replace(" ", "_")
        if value == "PRESENT":
            judgments.append(True)
        elif value == "NOT_PRESENT":
            judgments.append(False)
        else:
            raise ValueError(f"invalid RaR grader label for rubric {index}")
    return judgments


def validate_rar_metric_options(
    metric_options: dict[str, Any] | None, *, default_model: str
) -> None:
    resolve_judge_settings(metric_options, default_model=default_model)


def score_rar_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None,
    *,
    default_model: str,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=default_model)
    jobs: list[tuple[int, int, str, int]] = []
    layouts: list[int] = []
    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("RaR sample/output mismatch")
        rubrics = sample.data["rubrics"]
        for generation_idx, generation in enumerate(output.generations):
            jobs.append(
                (
                    sample_idx,
                    generation_idx,
                    build_grader_prompt(sample.data["prompt"], generation, rubrics),
                    len(rubrics),
                )
            )
        layouts.append(len(output.generations))

    def judge(job: tuple[int, int, str, int]) -> tuple[list[bool], str | None]:
        _, _, prompt, count = job
        messages = [{"role": "user", "content": prompt}]
        errors: list[str] = []
        for _ in range(NORMAL_FORMAT_ATTEMPTS):
            try:
                text = chat_completion(settings, messages)
            except (RuntimeError, ValueError) as exc:
                errors.append(str(exc))
                break
            try:
                return parse_presence_response(text, count), None
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))

        constraint = local_constraint_body(
            settings,
            json_schema={
                "type": "object",
                "properties": {
                    str(index): {
                        "type": "string",
                        "enum": ["PRESENT", "NOT_PRESENT"],
                    }
                    for index in range(1, count + 1)
                },
                "required": [str(index) for index in range(1, count + 1)],
                "additionalProperties": False,
            },
        )
        if constraint is not None:
            try:
                text = chat_completion(settings, messages, extra_body=constraint)
                return parse_presence_response(text, count), None
            except (RuntimeError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        error = errors[-1] if errors else "RaR grader produced no valid judgment"
        return [False] * count, error

    judged = parallel_map(judge, jobs, workers=settings.workers, desc="RaR judge")
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for sample, generation_count in zip(samples, layouts, strict=True):
        rubrics = sample.data["rubrics"]
        per_generation: list[dict[str, Any]] = []
        for _ in range(generation_count):
            judgments, error = judged[offset]
            offset += 1
            denominator = sum(float(rubric["points"]) for rubric in rubrics)
            achieved = sum(
                float(rubric["points"])
                for rubric, met in zip(rubrics, judgments, strict=True)
                if met
            )
            score = achieved / denominator
            family_met: dict[str, int] = defaultdict(int)
            family_total: dict[str, int] = defaultdict(int)
            for rubric, met in zip(rubrics, judgments, strict=True):
                family = str(rubric["family"])
                family_total[family] += 1
                family_met[family] += int(met)
            per_generation.append(
                {
                    "score": score,
                    "is_pass": score >= 0.5,
                    "parsed": {
                        str(index): "PRESENT" if met else "NOT_PRESENT"
                        for index, met in enumerate(judgments, start=1)
                    },
                    "meta": {
                        "criterion_count": len(rubrics),
                        "criteria_met": sum(judgments),
                        "judge_failed": error is not None,
                        "judge_error": error,
                        "family_met": dict(family_met),
                        "family_total": dict(family_total),
                        "family_scores": {
                            family: family_met[family] / family_total[family]
                            for family in family_total
                        },
                    },
                }
            )
        results.append(per_generation)
    return results


def aggregate_rar(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    del metric_options
    scores: list[float] = []
    family_met: dict[str, int] = defaultdict(int)
    family_total: dict[str, int] = defaultdict(int)
    criteria_met = criterion_count = judge_failures = 0
    for sample in sample_results:
        for record in sample.get("records", []):
            scores.append(float(record["score"]))
            meta = record.get("meta", {})
            criteria_met += int(meta.get("criteria_met", 0))
            criterion_count += int(meta.get("criterion_count", 0))
            judge_failures += int(bool(meta.get("judge_failed", False)))
            for family, value in meta.get("family_met", {}).items():
                family_met[str(family)] += int(value)
            for family, value in meta.get("family_total", {}).items():
                family_total[str(family)] += int(value)

    count = len(scores)
    metrics = {
        "score": sum(scores) / count if count else 0.0,
        "score:n_samples": float(count),
        "criterion_satisfaction_rate": (
            criteria_met / criterion_count if criterion_count else 0.0
        ),
        "judge_failure_rate": judge_failures / count if count else 0.0,
    }
    for family in FAMILY_NAMES:
        metrics[f"family/{family.lower()}"] = (
            family_met[family] / family_total[family]
            if family_total[family]
            else 0.0
        )
        metrics[f"family/{family.lower()}:n_criteria"] = float(
            family_total[family]
        )
    return metrics


__all__ = [
    "aggregate_rar",
    "build_grader_prompt",
    "build_rar_prompt",
    "load_rar_samples",
    "parse_presence_response",
    "score_rar_generations_batch",
    "validate_rar_metric_options",
]
