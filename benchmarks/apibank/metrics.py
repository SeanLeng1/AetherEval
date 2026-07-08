from typing import Any

from aethereval.backends import count_text_tokens
from aethereval.core.types import GenerationOutput, Sample
from aethereval.metrics.common import to_records
from benchmarks.apibank.scoring import (
    aggregate_scores,
    extract_think_content,
    parse_assistant_output,
    score_record,
)


PRIMARY_METRIC = "Overall"
REQUIRES_TOKENIZER = True


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    del sample, generation
    raise RuntimeError(
        "APIBank requires score_generations_batch so LengthReward can use tokenizer tokens."
    )


def _score_one(
    sample: Sample,
    generation: str,
    think_token_count: int | None,
) -> dict[str, Any]:
    thought, tool_calls = parse_assistant_output(generation)
    scored = score_record(
        {
            "data": {"answer": sample.gold},
            "raw_output": generation,
            "thought": thought,
            "tool_calls": tool_calls,
        },
        think_token_count,
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
            "think_token_count": scored["think_token_count"],
        },
    }


def score_generations_batch(
    samples: list[Sample],
    generation_outputs: list[GenerationOutput],
    metric_options: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    options = metric_options or {}
    tokenizer = options.get("_tokenizer")
    if tokenizer is None:
        raise RuntimeError("APIBank batch scoring requires a runtime tokenizer.")

    results: list[list[dict[str, Any]]] = []
    for sample, output in zip(samples, generation_outputs, strict=True):
        if sample.id != output.sample_id:
            raise ValueError(
                f"APIBank batch scoring sample/output mismatch: {sample.id} != {output.sample_id}"
            )
        per_generation: list[dict[str, Any]] = []
        for generation in output.generations:
            think_content = extract_think_content(generation)
            think_token_count = (
                count_text_tokens(think_content, tokenizer)
                if think_content is not None
                else None
            )
            per_generation.append(_score_one(sample, generation, think_token_count))
        results.append(per_generation)
    return results


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
            "think_token_count": record.meta["think_token_count"],
        }

    return aggregate_scores(scores)


__all__ = [
    "PRIMARY_METRIC",
    "REQUIRES_TOKENIZER",
    "aggregate",
    "score_generation",
    "score_generations_batch",
]
