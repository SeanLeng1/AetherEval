from typing import Any

from aethereval.core.types import Sample
from aethereval.metrics.common import to_records
from benchmarks.apibank.scoring import (
    aggregate_scores,
    parse_assistant_output,
    score_record,
)


PRIMARY_METRIC = "Overall"


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    thought, tool_calls = parse_assistant_output(generation)
    scored = score_record(
        {
            "data": {"answer": sample.gold},
            "raw_output": generation,
            "thought": thought,
            "tool_calls": tool_calls,
        }
    )
    correct_score = scored["correct_score"]
    score = float(correct_score) if isinstance(correct_score, (int, float)) else 0.0

    return {
        "score": score,
        "is_pass": correct_score == 1,
        "parsed": {
            "thought": thought,
            "tool_calls": tool_calls,
            "format_errors": scored["format_errors"],
        },
        "meta": {
            "level": int(sample.meta["level"]),
            "correct_score": correct_score,
            "loose_score": scored["loose_score"],
            "format_score": scored["format_score"],
            "length_score": scored["length_score"],
            "think_word_count": scored["think_word_count"],
        },
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = metric_options or {}
    n = int(options.get("n", 1))
    if n != 1:
        raise ValueError(
            "APIBank is a greedy single-generation benchmark; expected n=1."
        )

    scores: dict[str, dict[str, Any]] = {}
    for item in sample_results:
        records = to_records(item["records"])
        if not records:
            continue
        if len(records) != 1:
            raise ValueError(
                f"APIBank expected one record for sample {item['sample_id']}, "
                f"got {len(records)}"
            )

        record = records[0]
        scores[record.sample_id] = {
            "score": record.meta["correct_score"],
            "loose_score": record.meta["loose_score"],
            "format_score": record.meta["format_score"],
            "length_score": record.meta["length_score"],
            "think_word_count": record.meta["think_word_count"],
        }

    return aggregate_scores(scores)


__all__ = ["PRIMARY_METRIC", "aggregate", "score_generation"]
