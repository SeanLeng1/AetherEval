import re
from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationOutput, Sample
from aethereval.core.task_defaults import resolve_task_default_metrics
from aethereval.metrics.common import mean_stderr
from benchmark_utils.llm_judge import (
    chat_completion,
    judge_generations,
    judge_with_format_retries,
    parse_json_object,
    resolve_judge_settings,
)


PRIMARY_METRIC = "overall_score"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = str(
    resolve_task_default_metrics("writingbench").get("judge_model", "claude-sonnet-4-5")
)
EVALUATE_SYSTEM = (
    "You are an expert evaluator with extensive experience in evaluating response "
    "of given query."
)
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}
# The score precedes the reason, so it survives a reason that cannot be parsed.
_SCORE_BEFORE_REASON_RE = re.compile(
    r'\s*(?:```(?:json)?\s*)?\{\s*"score"\s*:\s*(10|[1-9])\s*,\s*"reason"\s*:\s*"(.*)',
    re.DOTALL,
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


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)

    def make_jobs(
        sample: Sample, output: GenerationOutput, generation: str
    ) -> list[tuple[str, str]]:
        del output
        response = _strip_thinking(generation)
        return [
            (
                str(criterion["name"]),
                EVALUATE_PROMPT.format(
                    query=sample.data["query"],
                    response=response,
                    criteria=criterion,
                ),
            )
            for criterion in sample.data["checklist"]
        ]

    def judge(job: tuple[str, str]) -> dict[str, Any]:
        name, prompt = job
        # A criterion scored 0 for a malformed judgment records the last parse error.
        last_error: BaseException | None = None

        def complete(*args: Any, **kwargs: Any) -> str:
            nonlocal last_error
            try:
                return chat_completion(*args, **kwargs)
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                raise

        def parse(text: str) -> dict[str, Any] | None:
            nonlocal last_error
            try:
                return _parse_grade(name, text)
            except (ValueError, TypeError) as exc:
                last_error = exc
                return None

        grade, last_text, _ = judge_with_format_retries(
            settings,
            [
                {"role": "system", "content": EVALUATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            parse,
            json_schema=GRADE_SCHEMA,
            complete=complete,
        )
        if grade is None:
            # A judge quoting a degenerate response can loop inside the reason until
            # the token limit; the official evaluator would abort the whole run here.
            grade = _recover_grade(name, last_text)
        if grade is None:
            grade = {
                "name": name,
                "score": 0,
                "reason": "",
                "raw": last_text,
                "judge_failed": True,
                "error": str(last_error),
            }
        return grade

    def combine(
        sample: Sample,
        output: GenerationOutput,
        gen_idx: int,
        criterion_grades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del sample, output, gen_idx
        score = sum(float(item["score"]) for item in criterion_grades) / len(
            criterion_grades
        )
        return {
            "score": score,
            "is_pass": score >= 5.0,
            "parsed": criterion_grades,
            "meta": {
                "criterion_scores": {
                    item["name"]: float(item["score"]) for item in criterion_grades
                }
            },
        }

    return judge_generations(
        samples,
        generation_outputs,
        label="WritingBench",
        make_jobs=make_jobs,
        run_job=judge,
        combine=combine,
        workers=settings.workers,
    )


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
            for dimension, names in (
                sample["meta"].get("requirement_criteria", {}).items()
            ):
                for name in names:
                    if name not in criterion_scores:
                        raise ValueError(
                            f"WritingBench criterion {name!r} is missing for "
                            f"sample {sample['sample_id']}"
                        )
                    requirement_c[str(dimension)].append(float(criterion_scores[name]))

    grades = [
        grade
        for sample in sample_results
        for record in sample.get("records", [])
        for grade in record.get("parsed") or []
    ]
    recovered = sum(bool(grade.get("recovered")) for grade in grades)
    failed = sum(bool(grade.get("judge_failed")) for grade in grades)
    metrics: dict[str, Any] = {
        "overall_raw_1_10": _mean(all_scores),
        "overall_score": _mean(all_scores) * 10.0,
        "overall_score_stderr": mean_stderr(all_scores) * 10.0,
    }
    for name, values in sorted(domain1.items()):
        metrics[f"domain1/{name}"] = _mean(values) * 10.0
    for name, values in sorted(domain2.items()):
        metrics[f"domain2/{name}"] = _mean(values) * 10.0
    for dimension in ("style", "format", "length"):
        metrics[f"requirement/{dimension}_R"] = _mean(requirement_r[dimension]) * 10.0
        metrics[f"requirement/{dimension}_C"] = _mean(requirement_c[dimension]) * 10.0
    warnings = []
    if recovered:
        warnings.append(
            f"{recovered} criterion scores were read from judge responses whose "
            "reason could not be parsed"
        )
    if failed:
        warnings.append(
            f"{failed} criteria were scored 0 because the judge returned no "
            "parseable score"
        )
    if warnings:
        metrics["__warnings__"] = warnings
    return metrics


def _strip_thinking(text: str) -> str:
    marker = "</think>\n\n"
    pos = text.find(marker)
    return text[pos + len(marker) :] if pos >= 0 else text


def _parse_grade(name: str, text: str) -> dict[str, Any]:
    parsed = parse_json_object(text)
    score = parsed.get("score")
    reason = parsed.get("reason")
    if not isinstance(score, int) or not 1 <= score <= 10:
        raise ValueError("judge score must be an integer from 1 to 10")
    if not isinstance(reason, str):
        raise ValueError("judge reason must be a string")
    return {"name": name, "score": score, "reason": reason, "raw": text}


def _recover_grade(name: str, text: str) -> dict[str, Any] | None:
    match = _SCORE_BEFORE_REASON_RE.match(text)
    if match is None:
        return None
    return {
        "name": name,
        "score": int(match.group(1)),
        "reason": match.group(2),
        "raw": text,
        "recovered": True,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
