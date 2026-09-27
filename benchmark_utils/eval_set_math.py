from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import GenerationRecord, Sample
from aethereval.metrics.common import aggregate_binary_results

from .data import load_hf, write_task_jsonl
from .math_scoring import score_with_math_verify


DATASET_NAME = "RLLab/eval-set"
DATASET_REVISION = "39f531062c1974c38e136ee3b9445683fa4d576b"
DATA_FILE = "data/eval.jsonl"
MATH_PROMPT_SUFFIX = (
    "\n\nPlease think step by step, and put your final answer within \\boxed{}."
)
# AIME rows store the bare answer and the problem without MATH_PROMPT_SUFFIX.
AIME_YEARS = {"aime24": 2024, "aime25": 2025}


def load_eval_set_math_samples(
    task_dir: Path, data_file: str = DATA_FILE, *, gold_field: str = "solution"
) -> list[Sample]:
    rows = read_jsonl(task_dir / data_file)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("eval-set math row must be a JSON object")

        sample_id = str(row["id"])
        problem = str(row["problem"]).strip()
        gold = str(row[gold_field]).strip()
        if not problem:
            raise ValueError(f"Empty problem for sample {sample_id}")
        if not gold:
            raise ValueError(f"Empty {gold_field} for sample {sample_id}")

        samples.append(
            Sample(
                id=sample_id,
                gold=gold,
                meta={
                    "source": row["source"],
                    "subset": row["subset"],
                },
                data={
                    "problem": problem,
                    "solution": gold,
                },
            )
        )
    return samples


def build_eval_set_math_prompt(sample: Sample, *, add_suffix: bool = False) -> str:
    problem = str(sample.data["problem"]).strip()
    return problem + MATH_PROMPT_SUFFIX if add_suffix else problem


def prepare_eval_set_math_dataset(subset: str, task_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for idx, row in enumerate(load_hf(DATASET_NAME, subset, "train", DATASET_REVISION)):
        if subset in AIME_YEARS:
            problem = str(row["problem"]).strip()
            if not problem.endswith(MATH_PROMPT_SUFFIX):
                raise ValueError(f"{subset} row {idx}: missing eval-set math prompt suffix")
            rows.append(
                {
                    "id": f"{subset}_{idx}",
                    # build_eval_set_math_prompt(add_suffix=True) adds it back exactly once.
                    "problem": problem[: -len(MATH_PROMPT_SUFFIX)],
                    "answer": str(row["solution"]),
                    "year": AIME_YEARS[subset],
                    "source": DATASET_NAME,
                    "subset": subset,
                }
            )
        else:
            rows.append(
                {
                    "id": f"{subset}_{idx}",
                    "problem": str(row["problem"]),
                    "solution": str(row["solution"]),
                    "source": DATASET_NAME,
                    "subset": subset,
                }
            )

    write_task_jsonl(task_dir, rows)


def score_generation(
    sample: Sample,
    generation: str,
    *,
    boxed_gold: bool = False,
    keep_units_fallback: bool = False,
) -> dict[str, Any]:
    score, pred_values, gold_values, warning = score_with_math_verify(
        str(sample.gold),
        generation,
        boxed_gold=boxed_gold,
        keep_units_fallback=keep_units_fallback,
    )

    parsed = {
        "prediction_extracted": pred_values,
        "gold_extracted": gold_values,
    }
    meta: dict[str, Any] = {
        "prediction_extracted": pred_values[0] if pred_values else None,
    }
    if warning:
        meta["warning"] = warning

    return {
        "score": score,
        "is_pass": bool(score >= 1.0),
        "parsed": parsed,
        "meta": meta,
    }


def _parsed_prediction_extracted(record: GenerationRecord) -> bool:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    extracted = parsed.get("prediction_extracted")
    return isinstance(extracted, list) and len(extracted) > 0


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float | list[str]]:
    metrics = aggregate_binary_results(
        sample_results,
        metric_options,
        parsed_flag_fn=_parsed_prediction_extracted,
    )
    # A gold that math-verify cannot extract scores every response 0; surface it
    # in summary.json instead of only in each record's meta.
    sample_ids_by_warning: dict[str, list[str]] = {}
    for item in sample_results:
        warnings = {record["meta"].get("warning") for record in item["records"]}
        for warning in sorted(filter(None, warnings)):
            sample_ids_by_warning.setdefault(warning, []).append(str(item["sample_id"]))
    if sample_ids_by_warning:
        metrics["__warnings__"] = [
            f"{len(ids)} samples scored 0 with warning '{warning}' "
            f"(e.g. {', '.join(ids[:5])})"
            for warning, ids in sample_ids_by_warning.items()
        ]
    return metrics
