import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from benchmark_utils.llm_judge import (
    chat_completion,
    parallel_map,
    resolve_judge_settings,
)


PRIMARY_METRIC = "OP"
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("llmeval_med").get("judge_model", "gpt-4o")
)
SYSTEM_MESSAGE = "You are a helpful assistant."
THRESHOLD = 4.0
CATEGORY_CODES = {
    "医疗知识": "MK",
    "医疗语言理解": "MLU",
    "医疗推理": "MR",
    "医疗安全伦理": "MSE",
    "医疗文本生成": "MTG",
}
PROMPTS = json.loads(
    (Path(__file__).resolve().parent / "data/judge_prompts.json").read_text(
        encoding="utf-8"
    )
)


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    options = metric_options or {}
    resolve_judge_settings(options, default_model=DEFAULT_JUDGE_MODEL)
    if int(options.get("judge_repeats", 3)) < 1:
        raise ValueError("judge_repeats must be >= 1")


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError("LLMEval-Med requires batched LLM-judge scoring")


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    options = metric_options or {}
    settings = resolve_judge_settings(options, default_model=DEFAULT_JUDGE_MODEL)
    repeats = int(options.get("judge_repeats", 3))
    jobs: list[tuple[int, int, int, str]] = []
    layouts: list[int] = []
    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("LLMEval-Med sample/output mismatch")
        prompt = _judge_prompt(sample, "")
        for gen_idx, generation in enumerate(output.generations):
            rendered = prompt.replace("<<Response>>", generation)
            for repeat_idx in range(repeats):
                jobs.append((sample_idx, gen_idx, repeat_idx, rendered))
        layouts.append(len(output.generations))

    def judge(job: tuple[int, int, int, str]) -> dict[str, Any]:
        _, _, _, prompt = job
        last = ""
        for _ in range(5):
            last = chat_completion(
                settings,
                [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
            match = re.search(r"\[(\d+)\]", last)
            if match and 1 <= int(match.group(1)) <= 5:
                return {"score": int(match.group(1)), "raw": last}
        raise RuntimeError(f"LLMEval-Med judge returned no [1-5] score: {last!r}")

    grades = parallel_map(judge, jobs, workers=settings.workers, desc="LLMEval-Med judge")
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for generation_count in layouts:
        per_sample: list[dict[str, Any]] = []
        for _ in range(generation_count):
            repeat_grades = grades[offset : offset + repeats]
            offset += repeats
            score = sum(float(item["score"]) for item in repeat_grades) / repeats
            per_sample.append(
                {
                    "score": score,
                    "is_pass": score >= THRESHOLD,
                    "parsed": repeat_grades,
                    "meta": {
                        "judge_scores": [item["score"] for item in repeat_grades]
                    },
                }
            )
        results.append(per_sample)
    return results


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del metric_options
    categories: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for sample in sample_results:
        category = str(sample["meta"]["category"])
        for record in sample.get("records", []):
            score = float(record["score"])
            categories[category].append(score)
            all_scores.append(score)

    metrics: dict[str, Any] = {}
    total_usable = 0
    total_count = 0
    for category, code in CATEGORY_CODES.items():
        scores = categories.get(category, [])
        usable = sum(score >= THRESHOLD for score in scores)
        metrics[f"{code}_usability_rate"] = usable / len(scores) * 100.0 if scores else 0.0
        metrics[f"{code}_avg_judge_score"] = _mean(scores)
        total_usable += usable
        total_count += len(scores)
    metrics["OP"] = total_usable / total_count * 100.0 if total_count else 0.0
    metrics["avg_judge_score"] = _mean(all_scores)
    metrics["__warnings__"] = [
        "MTG in the paper uses a five-dimension human evaluation with a safety veto. "
        "The released automated pipeline only supports the same averaged GPT-4o >=4 "
        "approximation reported here."
    ]
    return metrics


def _judge_prompt(sample: Sample, response: str) -> str:
    category = str(sample.data["category"])
    if category not in PROMPTS:
        raise ValueError(f"Unknown LLMEval-Med category: {category}")
    return (
        PROMPTS[category]
        .replace("<<Question>>", str(sample.data["problem"]))
        .replace("<<Sanswer>>", str(sample.data.get("sanswer")))
        .replace("<<checklist>>", str(sample.data.get("checklist")))
        .replace("<<Response>>", response or "<<Response>>")
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
