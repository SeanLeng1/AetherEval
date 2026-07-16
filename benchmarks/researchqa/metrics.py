import math
import random
from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from benchmark_utils.llm_judge import (
    chat_completion,
    parallel_map,
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


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError("ResearchQA requires batched LLM-judge scoring")


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)
    jobs: list[tuple[int, int, int, list[dict[str, Any]], str]] = []
    layouts: list[list[int]] = []
    batch_size = 8
    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("ResearchQA sample/output mismatch")
        per_generation: list[int] = []
        for gen_idx, generation in enumerate(output.generations):
            count = 0
            rubrics = sample.data["rubric"]
            for start in range(0, len(rubrics), batch_size):
                batch = rubrics[start : start + batch_size]
                prompt = _build_judge_prompt(
                    generation, [str(item["rubric_item"]) for item in batch]
                )
                jobs.append((sample_idx, gen_idx, start, batch, prompt))
                count += 1
            per_generation.append(count)
        layouts.append(per_generation)

    def judge(
        job: tuple[int, int, int, list[dict[str, Any]], str]
    ) -> list[dict[str, Any]]:
        _, _, _, rubrics, prompt = job
        last_output = ""
        for _ in range(3):
            last_output = chat_completion(
                settings,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            labels = [line.strip() for line in last_output.splitlines() if line.strip()]
            if len(labels) == len(rubrics) and all(label in LABELS for label in labels):
                return [
                    {
                        "rubric": rubric["rubric_item"],
                        "type": rubric.get("type", []),
                        "label": label,
                        "normalized_score": (LABELS[label] - 1) / 4.0,
                    }
                    for rubric, label in zip(rubrics, labels, strict=True)
                ]
        raise RuntimeError(
            "ResearchQA judge returned the wrong number or kind of labels: "
            f"{last_output!r}"
        )

    batches = parallel_map(judge, jobs, workers=settings.workers, desc="ResearchQA judge")
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for per_generation in layouts:
        per_sample: list[dict[str, Any]] = []
        for batch_count in per_generation:
            rubric_grades: list[dict[str, Any]] = []
            for batch in batches[offset : offset + batch_count]:
                rubric_grades.extend(batch)
            offset += batch_count
            score = sum(item["normalized_score"] for item in rubric_grades) / len(
                rubric_grades
            )
            per_sample.append(
                {
                    "score": score,
                    "is_pass": score >= 0.5,
                    "parsed": rubric_grades,
                    "meta": {"rubric_grades": rubric_grades},
                }
            )
        results.append(per_sample)
    return results


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    options = metric_options or {}
    values: list[float] = []
    domains: dict[str, list[float]] = defaultdict(list)
    fields: dict[str, list[float]] = defaultdict(list)
    for sample in sample_results:
        for record in sample.get("records", []):
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
    }
    for name, scores in sorted(domains.items()):
        metrics[f"domain/{name}"] = _mean(scores) * 100.0
    for name, scores in sorted(fields.items()):
        metrics[f"field/{name}"] = _mean(scores) * 100.0
    return metrics


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
        "Questions:\n"
        + "\n".join(questions)
        + "\n\nOutput:"
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
