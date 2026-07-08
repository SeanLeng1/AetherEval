import json
from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import GenerationRecord, Sample
from aethereval.metrics.common import aggregate_binary_results

from .math_scoring import score_with_math_verify


DATASET_NAME = "RLLab/eval-set"
DATA_FILE = "data/eval.jsonl"


def load_eval_set_math_samples(
    task_dir: Path, data_file: str = DATA_FILE
) -> list[Sample]:
    rows = read_jsonl(task_dir / data_file)
    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("eval-set math row must be a JSON object")

        sample_id = str(row["id"])
        problem = str(row["problem"]).strip()
        solution = str(row["solution"]).strip()
        if not problem:
            raise ValueError(f"Empty problem for sample {sample_id}")
        if not solution:
            raise ValueError(f"Empty solution for sample {sample_id}")

        samples.append(
            Sample(
                id=sample_id,
                gold=solution,
                meta={
                    "source": row["source"],
                    "subset": row["subset"],
                },
                data={
                    "problem": problem,
                    "solution": solution,
                },
            )
        )
    return samples


def build_eval_set_math_prompt(sample: Sample) -> str:
    return str(sample.data["problem"]).strip()


def prepare_eval_set_math_dataset(subset: str, task_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for prepare_data.py. Install with `pip install datasets`."
        ) from exc

    out_path = task_dir / DATA_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(load_dataset(DATASET_NAME, subset, split="train")):
        rows.append(
            {
                "id": f"{subset}_{idx}",
                "problem": str(row["problem"]),
                "solution": str(row["solution"]),
                "source": DATASET_NAME,
                "subset": subset,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    score, pred_values, gold_values, warning = score_with_math_verify(
        str(sample.gold),
        generation,
        boxed_gold=False,
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
