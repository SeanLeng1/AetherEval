import random
import re
from pathlib import Path
from typing import Any

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from benchmark_utils.llm_judge import (
    NORMAL_FORMAT_ATTEMPTS,
    chat_completion,
    local_constraint_body,
    parallel_map,
    parse_json_object,
    resolve_judge_settings,
)


PRIMARY_METRIC = "eqbench_creative_score"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("creative_writing_v3").get(
        "judge_model", "claude-sonnet-4-6"
    )
)
ROOT = Path(__file__).resolve().parent
CRITERIA = [
    line.strip()
    for line in (ROOT / "criteria.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
NEGATIVE_CRITERIA = [
    line.strip()
    for line in (ROOT / "negative_criteria.txt")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
NEGATIVE_CRITERIA_SET = set(NEGATIVE_CRITERIA)
JUDGE_PROMPT = (ROOT / "judge_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        criterion: {"type": "number", "minimum": 0, "maximum": 20}
        for criterion in CRITERIA
    },
    "additionalProperties": False,
    "minProperties": 1,
}


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError("Creative Writing Bench V3 requires batched LLM-judge scoring")


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)
    jobs: list[tuple[int, int, str]] = []
    layouts: list[int] = []
    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("creative_writing_v3 sample/output mismatch")
        for gen_idx, generation in enumerate(output.generations):
            if (
                output.meta.get("creative_generation_failed")
                or len(generation.strip()) < 500
            ):
                jobs.append((sample_idx, gen_idx, ""))
                continue
            prompt = JUDGE_PROMPT.format(
                writing_prompt=sample.data["base_prompt"],
                test_model_response=generation,
                creative_writing_criteria="\n".join(f"- {item}" for item in CRITERIA),
                lower_is_better_criteria=", ".join(NEGATIVE_CRITERIA),
            )
            jobs.append((sample_idx, gen_idx, prompt))
        layouts.append(len(output.generations))

    def judge(job: tuple[int, int, str]) -> dict[str, Any]:
        _, _, prompt = job
        if not prompt:
            return {"scores": {}, "raw": "", "generation_failed": True}
        last_text = ""
        messages = [{"role": "user", "content": prompt}]
        for _ in range(NORMAL_FORMAT_ATTEMPTS):
            last_text = chat_completion(
                settings,
                messages,
            )
            scores = _parse_scores(last_text)
            if scores:
                return {
                    "scores": scores,
                    "raw": last_text,
                    "generation_failed": False,
                }

        constraint = local_constraint_body(settings, json_schema=SCORE_SCHEMA)
        if constraint is not None:
            try:
                last_text = chat_completion(
                    settings,
                    messages,
                    extra_body=constraint,
                )
                scores = _parse_scores(last_text)
                if scores:
                    return {
                        "scores": scores,
                        "raw": last_text,
                        "generation_failed": False,
                    }
            except (RuntimeError, ValueError):
                pass

        # Upstream leaves the task unjudged and excludes it from aggregation.
        return {
            "scores": {},
            "raw": last_text,
            "generation_failed": False,
            "error": "judge returned no parseable scores",
        }

    judged = parallel_map(
        judge, jobs, workers=settings.workers, desc="Creative Writing V3 judge"
    )
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for count in layouts:
        per_sample: list[dict[str, Any]] = []
        for item in judged[offset : offset + count]:
            offset += 1
            scores = item["scores"]
            adjusted = [
                20.0 - value if name in NEGATIVE_CRITERIA_SET else value
                for name, value in scores.items()
                if isinstance(value, (int, float)) and 0.0 <= value <= 20.0
            ]
            piece_score = sum(adjusted) / len(adjusted) if adjusted else 0.0
            per_sample.append(
                {
                    "score": piece_score,
                    "is_pass": bool(adjusted) and piece_score >= 10.0,
                    "parsed": item,
                    "meta": {
                        "generation_failed": bool(item["generation_failed"]),
                        "judge_failed": "error" in item,
                        "_aethereval_unscored": "error" in item,
                        "judge_scores": scores,
                    },
                }
            )
        results.append(per_sample)
    return results


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = metric_options or {}
    scores: list[float] = []
    generation_failures = 0
    judge_failures = 0
    for sample in sample_results:
        for record in sample.get("records", []):
            if record.get("meta", {}).get("generation_failed"):
                generation_failures += 1
                continue
            if record.get("meta", {}).get("judge_failed"):
                judge_failures += 1
                continue
            scores.append(float(record["score"]))
    raw = _mean(scores)
    metrics: dict[str, Any] = {
        "creative_score_0_20": round(raw, 2),
        "eqbench_creative_score": round(raw * 5.0, 2),
        "scored_pieces": float(len(scores)),
        "generation_failures": float(generation_failures),
        "judge_failures": float(judge_failures),
    }
    count = int(options.get("bootstrap_resamples", 1000))
    if scores and count > 0:
        rng = random.Random(int(options.get("bootstrap_seed", 42)))
        boot = sorted(_mean([rng.choice(scores) for _ in scores]) for _ in range(count))
        confidence = float(options.get("bootstrap_confidence", 0.95))
        lower = int((1.0 - confidence) / 2.0 * len(boot))
        upper = min(len(boot) - 1, int((1.0 + confidence) / 2.0 * len(boot)) - 1)
        metrics["creative_score_ci_lower"] = boot[max(0, lower)]
        metrics["creative_score_ci_upper"] = boot[upper]
    warnings = []
    if generation_failures:
        warnings.append(
            f"{generation_failures} pieces remained shorter than 500 characters "
            "after 3 attempts and were excluded, matching upstream behavior."
        )
    if judge_failures:
        warnings.append(
            f"{judge_failures} pieces had no parseable judge scores and were "
            "excluded, matching upstream behavior."
        )
    if warnings:
        metrics["__warnings__"] = warnings
    return metrics


def _parse_scores(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    try:
        parsed = parse_json_object(text)
    except (ValueError, TypeError):
        parsed = {}
    for name, raw in parsed.items():
        if name in SCORE_SCHEMA["properties"] and isinstance(raw, (int, float)):
            value = float(raw)
            if 0.0 <= value <= 20.0:
                scores[name] = value
    if scores:
        return scores

    patterns = (
        r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)",
        r"(.*?):\s*\[(-?\d+(?:\.\d+)?)\]",
    )
    for pattern in patterns:
        for name, raw in re.findall(pattern, text):
            value = float(raw)
            if value <= 20.0:
                scores[name.strip()] = value
    return scores


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
