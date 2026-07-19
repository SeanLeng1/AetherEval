import math
import random
from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from benchmark_utils.llm_judge import (
    chat_completion,
    parallel_map,
    parse_json_object,
    resolve_judge_settings,
)


PRIMARY_METRIC = "score"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("healthbench").get(
        "judge_model", "gpt-4.1-2025-04-14"
    )
)
SYSTEM_MESSAGE = "You are a helpful assistant."

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError("HealthBench requires batched LLM-judge scoring")


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    options = metric_options or {}
    settings = resolve_judge_settings(options, default_model=DEFAULT_JUDGE_MODEL)
    jobs: list[tuple[int, int, int, str]] = []
    layouts: list[list[int]] = []

    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("HealthBench sample/output mismatch")
        per_generation: list[int] = []
        for gen_idx, generation in enumerate(output.generations):
            conversation = sample.data["prompt"] + [
                {"role": "assistant", "content": generation}
            ]
            convo_str = "\n\n".join(
                f"{message['role']}: {message['content']}" for message in conversation
            )
            count = 0
            for rubric_idx, rubric in enumerate(sample.data["rubrics"]):
                rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
                prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace(
                    "<<rubric_item>>", rubric_text
                )
                jobs.append((sample_idx, gen_idx, rubric_idx, prompt))
                count += 1
            per_generation.append(count)
        layouts.append(per_generation)

    def judge(job: tuple[int, int, int, str]) -> dict[str, Any]:
        _, _, _, prompt = job
        last_error: BaseException | None = None
        for _ in range(20):
            text = chat_completion(
                settings,
                [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
            )
            try:
                parsed = parse_json_object(text)
                if parsed.get("criteria_met") is True or parsed.get("criteria_met") is False:
                    return {
                        "criteria_met": bool(parsed["criteria_met"]),
                        "explanation": str(parsed.get("explanation", "")),
                        "raw": text,
                    }
            except (ValueError, TypeError) as exc:
                last_error = exc
        raise RuntimeError(f"HealthBench judge returned invalid JSON: {last_error}")

    grades = parallel_map(judge, jobs, workers=settings.workers, desc="HealthBench judge")
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for sample, per_generation in zip(samples, layouts, strict=True):
        sample_results: list[dict[str, Any]] = []
        for rubric_count in per_generation:
            rubric_grades = grades[offset : offset + rubric_count]
            offset += rubric_count
            rubrics = sample.data["rubrics"]
            positive_total = sum(float(r["points"]) for r in rubrics if float(r["points"]) > 0)
            achieved = sum(
                float(rubric["points"])
                for rubric, grade in zip(rubrics, rubric_grades, strict=True)
                if grade["criteria_met"]
            )
            score = achieved / positive_total
            tag_values: dict[str, float] = {
                str(tag): score for tag in sample.data["example_tags"]
            }
            tagged: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
            for rubric, grade in zip(rubrics, rubric_grades, strict=True):
                for tag in rubric.get("tags", []):
                    tagged[str(tag)].append((rubric, grade))
            for tag, pairs in tagged.items():
                denom = sum(
                    float(rubric["points"])
                    for rubric, _ in pairs
                    if float(rubric["points"]) > 0
                )
                if denom:
                    tag_values[tag] = sum(
                        float(rubric["points"])
                        for rubric, grade in pairs
                        if grade["criteria_met"]
                    ) / denom
            sample_results.append(
                {
                    "score": score,
                    "is_pass": score >= 0.5,
                    "parsed": rubric_grades,
                    "meta": {"tag_scores": tag_values},
                }
            )
        results.append(sample_results)
    return results


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    options = metric_options or {}
    values: list[float] = []
    tags: dict[str, list[float]] = defaultdict(list)
    for sample in sample_results:
        for record in sample.get("records", []):
            values.append(float(record["score"]))
            for tag, score in record.get("meta", {}).get("tag_scores", {}).items():
                tags[str(tag)].append(float(score))

    metrics: dict[str, float] = {
        "score": _clipped_mean(values),
        "score:n_samples": float(len(values)),
        "score:bootstrap_std": _bootstrap_std(
            values,
            int(options.get("bootstrap_resamples", 1000)),
            int(options.get("bootstrap_seed", 42)),
        ),
    }
    for tag, tag_values in sorted(tags.items()):
        metrics[tag] = _clipped_mean(tag_values)
        metrics[f"{tag}:n_samples"] = float(len(tag_values))
        metrics[f"{tag}:bootstrap_std"] = _bootstrap_std(
            tag_values,
            int(options.get("bootstrap_resamples", 1000)),
            int(options.get("bootstrap_seed", 42)),
        )
    return metrics


def _clipped_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return min(1.0, max(0.0, sum(values) / len(values)))


def _bootstrap_std(values: list[float], count: int, seed: int) -> float:
    if not values or count <= 0:
        return 0.0
    rng = random.Random(seed)
    means = [
        _clipped_mean([rng.choice(values) for _ in values]) for _ in range(count)
    ]
    mean_value = sum(means) / len(means)
    return math.sqrt(sum((value - mean_value) ** 2 for value in means) / len(means))
