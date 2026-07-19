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


PRIMARY_METRIC = "overall_score"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("writingbench").get(
        "judge_model", "claude-sonnet-4-5"
    )
)
EVALUATE_SYSTEM = (
    "You are an expert evaluator with extensive experience in evaluating response "
    "of given query."
)
EVALUATE_PROMPT = """
Evaluate the Response based on the Query and Criteria provided following the Scoring Rules.

** Scoring Rules **

"1-2": "Low score description: Critical deficiencies and major issues that prevent adequate functionality.",
"3-4": "Below average score description: Lacking with noticeable shortcomings that impact overall effectiveness and require improvement.",
"5-6": "Average score description: Adequate but not exemplary, Baseline performance that meets essential requirements. Most models may achieve this score.",
"7-8": "Above average score description: Strong performance characterized by competent execution, though minor refinements are needed to achieve excellence.",
"9-10": "High score description: Exceptional performance with all aspects optimally addressed, demonstrating superior effectiveness and quality without any flaws."

-Provide reasons for each score by indicating specific strengths or deficiencies within the Response. Reference exact text passages to justify the score, ensuring that each reason is concrete and aligns with the criteria requirements while highlighting key gaps from the ideal answer.

-Be very STRICT and do not be misled by format or length; ensure that the Response is thoroughly evaluated beyond superficial appearances.

-Carefully discern whether the content of the Response is an illusion, appearing substantial but actually entirely fabricated.

-Sometimes the model may only provide an introduction or an overview without truly completing the query, which should be considered a failed response. Carefully discern this.

-Scoring Range: Assign an integer score between 1 to 10

** Output format ** 
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}

** Criteria **
```{criteria}```

** Query **
```{query}```

** Response **
```{response}```

Provide your evaluation based on the criteria restated below:

```{criteria}```

** Output format ** 
(Remove symbols that interfere with JSON parsing, don't use " inside reason)
Return the results in the following JSON format, Only output the following JSON format and nothing else:
```json
{{
    "score": an integer score between 1 to 10,
    "reason": "Specific and detailed justification for the score using text elements."
}}
```
""".strip()


def validate_metric_options(metric_options: dict[str, Any] | None = None) -> None:
    resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError("WritingBench requires batched LLM-judge scoring")


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)
    jobs: list[tuple[int, int, int, str, str]] = []
    layouts: list[list[int]] = []
    for sample_idx, (sample, output) in enumerate(
        zip(samples, generation_outputs, strict=True)
    ):
        if sample.id != output.sample_id:
            raise ValueError("WritingBench sample/output mismatch")
        per_generation: list[int] = []
        for gen_idx, generation in enumerate(output.generations):
            response = _strip_thinking(generation)
            for criterion_idx, criterion in enumerate(sample.data["checklist"]):
                prompt = EVALUATE_PROMPT.format(
                    query=sample.data["query"],
                    response=response,
                    criteria=criterion,
                )
                jobs.append(
                    (sample_idx, gen_idx, criterion_idx, str(criterion["name"]), prompt)
                )
            per_generation.append(len(sample.data["checklist"]))
        layouts.append(per_generation)

    def judge(job: tuple[int, int, int, str, str]) -> dict[str, Any]:
        _, _, _, name, prompt = job
        last_error: BaseException | None = None
        for _ in range(15):
            text = chat_completion(
                settings,
                [
                    {"role": "system", "content": EVALUATE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            try:
                parsed = parse_json_object(text)
                score = parsed.get("score")
                reason = parsed.get("reason")
                if isinstance(score, int) and 1 <= score <= 10 and isinstance(reason, str):
                    return {"name": name, "score": score, "reason": reason, "raw": text}
            except (ValueError, TypeError) as exc:
                last_error = exc
        raise RuntimeError(f"WritingBench judge returned invalid score: {last_error}")

    grades = parallel_map(judge, jobs, workers=settings.workers, desc="WritingBench judge")
    results: list[list[dict[str, Any]]] = []
    offset = 0
    for per_generation in layouts:
        per_sample: list[dict[str, Any]] = []
        for count in per_generation:
            criterion_grades = grades[offset : offset + count]
            offset += count
            score = sum(float(item["score"]) for item in criterion_grades) / count
            per_sample.append(
                {
                    "score": score,
                    "is_pass": score >= 5.0,
                    "parsed": criterion_grades,
                    "meta": {
                        "criterion_scores": {
                            item["name"]: float(item["score"]) for item in criterion_grades
                        }
                    },
                }
            )
        results.append(per_sample)
    return results


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    del metric_options
    all_scores: list[float] = []
    domain1: dict[str, list[float]] = defaultdict(list)
    domain2: dict[str, list[float]] = defaultdict(list)
    requirement_r: dict[str, list[float]] = defaultdict(list)
    requirement_c: dict[str, list[float]] = defaultdict(list)
    for sample in sample_results:
        for record in sample.get("records", []):
            score = float(record["score"])
            all_scores.append(score)
            domain1[str(sample["meta"]["domain1"])].append(score)
            domain2[str(sample["meta"]["domain2"])].append(score)
            for dimension in sample["meta"].get("requirement_subsets", []):
                requirement_r[str(dimension)].append(score)
            criterion_scores = record.get("meta", {}).get("criterion_scores", {})
            for dimension, names in sample["meta"].get(
                "requirement_criteria", {}
            ).items():
                for name in names:
                    if name not in criterion_scores:
                        raise ValueError(
                            f"WritingBench criterion {name!r} is missing for "
                            f"sample {sample['sample_id']}"
                        )
                    requirement_c[str(dimension)].append(
                        float(criterion_scores[name])
                    )

    metrics: dict[str, float] = {
        "overall_raw_1_10": _mean(all_scores),
        "overall_score": _mean(all_scores) * 10.0,
    }
    for name, values in sorted(domain1.items()):
        metrics[f"domain1/{name}"] = _mean(values) * 10.0
    for name, values in sorted(domain2.items()):
        metrics[f"domain2/{name}"] = _mean(values) * 10.0
    for dimension in ("style", "format", "length"):
        metrics[f"requirement/{dimension}_R"] = (
            _mean(requirement_r[dimension]) * 10.0
        )
        metrics[f"requirement/{dimension}_C"] = (
            _mean(requirement_c[dimension]) * 10.0
        )
    return metrics


def _strip_thinking(text: str) -> str:
    marker = "</think>\n\n"
    pos = text.find(marker)
    return text[pos + len(marker) :] if pos >= 0 else text


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
