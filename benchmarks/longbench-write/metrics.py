import re
from collections import defaultdict
from pathlib import Path

from aethereval.core.task_defaults import resolve_task_default_metrics
from aethereval.metrics.common import mean
from benchmark_utils.llm_judge import (
    judge_generations,
    judge_with_format_retries,
    parse_json_object,
    resolve_judge_settings,
)

PRIMARY_METRIC = "overall_score"
USES_LLM_JUDGE = True
PRESERVE_EXISTING_SCORES_ON_RESUME = True
DEFAULT_JUDGE_MODEL = resolve_task_default_metrics("longbench-write")["judge_model"]
DIMENSIONS = (
    "Relevance",
    "Accuracy",
    "Coherence",
    "Clarity",
    "Breadth and Depth",
    "Reading Experience",
)
PROMPT = Path(__file__).with_name("judge.txt").read_text(encoding="utf-8")
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "Analysis": {"type": "string"},
        **{
            name: {"type": "integer", "minimum": 1, "maximum": 5} for name in DIMENSIONS
        },
    },
    "required": ["Analysis", *DIMENSIONS],
    "additionalProperties": False,
}


def count_words(text):
    # Verbatim counting convention in upstream evaluation/pred.py.
    return len(re.findall(r"[\u4e00-\u9fff]", text)) + len(
        re.findall(r"\b[a-zA-Z]+\b", text)
    )


def length_score(requested, actual):
    if requested <= 0:
        raise ValueError("Requested word count must be positive")
    if actual <= 0:
        return 0.0  # Continuous zero-output limit; upstream divides by zero.
    if actual > requested:
        return 100 * max(0, 1 - (actual / requested - 1) / 3)
    return 100 * max(0, 1 - (requested / actual - 1) / 2)


def parse_grade(text):
    try:
        grade = parse_json_object(text)
    except (ValueError, TypeError):
        return None
    if any(
        type(grade.get(name)) is not int or not 1 <= grade[name] <= 5
        for name in DIMENSIONS
    ):
        return None
    return grade


def validate_metric_options(metric_options=None):
    resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)


def score_generations_batch(samples, outputs, metric_options=None):
    settings = resolve_judge_settings(metric_options, default_model=DEFAULT_JUDGE_MODEL)

    def make_jobs(sample, output, generation):
        del output
        return [
            PROMPT.replace("$INST$", sample.data["prompt"]).replace(
                "$RESPONSE$", generation
            )
        ]

    def judge(prompt):
        grade, text, attempts = judge_with_format_retries(
            settings,
            [{"role": "user", "content": prompt}],
            parse_grade,
            json_schema=GRADE_SCHEMA,
            error_as_text=True,
        )
        return {"grade": grade, "raw": text, "attempts": attempts}

    def combine(sample, output, gen_idx, results):
        actual = count_words(output.generations[gen_idx])
        sl = length_score(sample.data["length"], actual)
        result = results[0]
        grade = result["grade"]
        sq = (
            (mean([grade[name] for name in DIMENSIONS]) - 1) * 25
            if grade is not None
            else None
        )
        return {
            "score": (sq + sl) / 2 if sq is not None else 0,
            "parsed": {
                "quality_score": sq,
                "length_score": sl,
                "response_words": actual,
                **result,
            },
            "meta": {
                "_aethereval_unscored": grade is None,
                "judge_failed": grade is None,
            },
        }

    return judge_generations(
        samples,
        outputs,
        label="LongBench-Write",
        make_jobs=make_jobs,
        run_job=judge,
        combine=combine,
        workers=settings.workers,
    )


def aggregate(sample_results, metric_options=None):
    del metric_options
    quality, lengths, overall, words, dims = [], [], [], [], defaultdict(list)
    grouped = defaultdict(list)
    failed = 0
    for item in sample_results:
        for record in item["records"]:
            parsed = record.get("parsed") or {}
            if parsed.get("length_score") is not None:
                lengths.append(parsed["length_score"])
                words.append(parsed["response_words"])
            if parsed.get("quality_score") is None:
                failed += 1
                continue
            quality.append(parsed["quality_score"])
            overall.append(record["score"])
            grouped[item["meta"]["language"]].append(record["score"])
            for name in DIMENSIONS:
                dims[name].append((parsed["grade"][name] - 1) * 25)
    return {
        "overall_score": mean(overall),
        "quality_score": mean(quality),
        "length_score": mean(lengths),
        "avg_response_words": mean(words),
        "failed_judgments": failed,
        **{f"quality/{name}": mean(values) for name, values in dims.items()},
        **{f"language/{name}": mean(values) for name, values in grouped.items()},
    }
