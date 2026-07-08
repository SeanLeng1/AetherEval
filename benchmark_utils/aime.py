import json
from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import GenerationRecord, Sample
from aethereval.metrics.common import aggregate_binary_results

from .math_scoring import score_with_math_verify


MATH_PROMPT_TEMPLATE = (
    "{Question}\n\n"
    "Please think step by step, and put your final answer within \\boxed{{}}."
)


def load_aime_samples(task_dir: Path, data_file: str) -> list[Sample]:
    rows = read_jsonl(task_dir / data_file)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("AIME row must be a JSON object")
        sample_id = str(row["id"])
        problem = str(row["problem"]).strip()
        answer = str(row["answer"]).strip()
        if not problem:
            raise ValueError(f"Empty problem for sample {sample_id}")
        if not answer:
            raise ValueError(f"Empty answer for sample {sample_id}")
        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta={
                    "year": row.get("year"),
                    "url": row.get("url"),
                },
                data={
                    "problem": problem,
                    "solution": row.get("solution"),
                },
            )
        )
    return samples


def build_aime_prompt(sample: Sample) -> str:
    return MATH_PROMPT_TEMPLATE.format(Question=str(sample.data["problem"]))


def prepare_aime_dataset(dataset_name: str, task_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for prepare_data.py. Install with `pip install datasets`."
        ) from exc

    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for row in load_dataset(dataset_name, "default", split="train"):
        rows.append(
            {
                "id": str(row["id"]),
                "problem": str(row["problem"]),
                "answer": str(row["answer"]),
                "solution": str(row.get("solution", "")),
                "url": row.get("url"),
                "year": row.get("year"),
                "source": dataset_name,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    gold = str(sample.gold).strip()
    score, pred_values, gold_values, warning = score_with_math_verify(
        gold,
        generation,
        boxed_gold=True,
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
    return aggregate_binary_results(
        sample_results,
        metric_options,
        parsed_flag_fn=_parsed_prediction_extracted,
    )
