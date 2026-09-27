import math
import random
from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from benchmark_utils.llm_judge import (
    chat_completion,
    judge_generations,
    judge_with_format_retries,
    resolve_judge_settings,
)


PRIMARY_METRIC = "coverage"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("researchqa").get("judge_model", "gpt-4.1-mini")
)
LABELS = {
    "Not at all": 1,
    "Barely": 2,
    "Moderately": 3,
    "Mostly": 4,
    "Completely": 5,
}


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)
    batch_size = 8
    label_pattern = "(" + "|".join(LABELS) + ")"

    def make_jobs(
        sample: Sample, output: GenerationOutput, generation: str
    ) -> list[tuple[list[dict[str, Any]], str]]:
        del output
        rubrics = sample.data["rubric"]
        jobs: list[tuple[list[dict[str, Any]], str]] = []
        for start in range(0, len(rubrics), batch_size):
            batch = rubrics[start : start + batch_size]
            prompt = _build_judge_prompt(
                generation, [str(item["rubric_item"]) for item in batch]
            )
            jobs.append((batch, prompt))
        return jobs

    def judge(job: tuple[list[dict[str, Any]], str]) -> list[dict[str, Any]]:
        rubrics, prompt = job
        parsed, last_output, _ = judge_with_format_retries(
            settings,
            [{"role": "user", "content": prompt}],
            lambda text: _parse_batch(rubrics, text),
            regex=r"\n".join(label_pattern for _ in rubrics),
            complete=chat_completion,
            error_as_text=True,
        )
        if parsed is not None:
            return parsed

        # The official ResearchQA evaluator skips the whole item when any
        # rubric batch still has the wrong shape after its retries.
        return [
            {
                "rubric": rubric["rubric_item"],
                "type": rubric.get("type", []),
                "label": None,
                "normalized_score": 0.0,
                "raw": last_output,
                "error": "judge returned the wrong number or kind of labels",
            }
            for rubric in rubrics
        ]

    def combine(
        sample: Sample,
        output: GenerationOutput,
        gen_idx: int,
        batches: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        del sample, output, gen_idx
        rubric_grades = [item for batch in batches for item in batch]
        judge_failed = any("error" in item for item in rubric_grades)
        score = (
            0.0
            if judge_failed
            else sum(item["normalized_score"] for item in rubric_grades)
            / len(rubric_grades)
        )
        return {
            "score": score,
            "is_pass": score >= 0.5,
            "parsed": rubric_grades,
            "meta": {
                "rubric_grades": rubric_grades,
                "judge_format_failures": sum("error" in item for item in rubric_grades),
                # compute_coverage.py skips such an item and averages the
                # rest; aggregate() does the same instead of aborting the run.
                "judge_failed": judge_failed,
            },
        }

    return judge_generations(
        samples,
        generation_outputs,
        label="ResearchQA",
        make_jobs=make_jobs,
        run_job=judge,
        combine=combine,
        workers=settings.workers,
    )


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    options = metric_options or {}
    values: list[float] = []
    domains: dict[str, list[float]] = defaultdict(list)
    fields: dict[str, list[float]] = defaultdict(list)
    judge_failures = 0
    for sample in sample_results:
        for record in sample.get("records", []):
            if record.get("meta", {}).get("judge_failed"):
                judge_failures += 1
                continue
            score = float(record["score"])
            values.append(score)
            domains[str(sample["meta"]["general_domain"])].append(score)
            fields[str(sample["meta"]["field"])].append(score)

    metrics: dict[str, float] = {
        "coverage": _mean(values) * 100.0,
        "coverage_bootstrap_std": _bootstrap_std(
            values,
            int(options.get("bootstrap_resamples", 1000)),
            int(options.get("bootstrap_seed", 42)),
        )
        * 100.0,
        "scored_samples": float(len(values)),
        "judge_failures": float(judge_failures),
    }
    for name, scores in sorted(domains.items()):
        metrics[f"domain/{name}"] = _mean(scores) * 100.0
    for name, scores in sorted(fields.items()):
        metrics[f"field/{name}"] = _mean(scores) * 100.0
    return metrics


def _parse_batch(
    rubrics: list[dict[str, Any]],
    text: str,
) -> list[dict[str, Any]] | None:
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    if len(labels) != len(rubrics) or any(label not in LABELS for label in labels):
        return None
    return [
        {
            "rubric": rubric["rubric_item"],
            "type": rubric.get("type", []),
            "label": label,
            "normalized_score": (LABELS[label] - 1) / 4.0,
        }
        for rubric, label in zip(rubrics, labels, strict=True)
    ]


def _build_judge_prompt(response: str, questions: list[str]) -> str:
    return (
        "Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n"
        f"Response: {response}\n"
        "Questions:\n" + "\n".join(questions) + "\n\nOutput:"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_std(values: list[float], count: int, seed: int) -> float:
    if not values or count <= 0:
        return 0.0
    rng = random.Random(seed)
    means = [_mean([rng.choice(values) for _ in values]) for _ in range(count)]
    center = _mean(means)
    return math.sqrt(sum((value - center) ** 2 for value in means) / len(means))
