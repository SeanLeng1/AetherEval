import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

from aethereval.core.types import Sample
from aethereval.metrics.common import mean, mean_stderr, to_records

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _gold_aliases(gold: Any) -> list[str]:
    if isinstance(gold, str):
        values = [gold]
    elif isinstance(gold, Sequence):
        values = [str(value) for value in gold]
    else:
        raise TypeError("Open-QA gold answers must be a string or sequence of aliases")
    aliases = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not aliases:
        raise ValueError("Open-QA sample has no nonempty gold aliases")
    return aliases


def _token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_open_qa(sample: Sample, generation: str) -> dict[str, Any]:
    # Reference NQ-Open and TriviaQA evaluation scores the submitted answer
    # itself.  Silently extracting a later ``Final answer:`` line would make
    # exact match more permissive than those protocols.
    prediction = str(generation).strip()
    aliases = _gold_aliases(sample.gold)
    prediction_normalized = normalize_answer(prediction)
    aliases_normalized = [normalize_answer(alias) for alias in aliases]
    exact_match = float(
        bool(prediction_normalized) and prediction_normalized in aliases_normalized
    )
    token_f1 = max(_token_f1(prediction, alias) for alias in aliases)
    return {
        "score": exact_match,
        "is_pass": bool(exact_match),
        "parsed": {
            "prediction": prediction,
            "prediction_normalized": prediction_normalized,
            "exact_match": exact_match,
            "token_f1": token_f1,
        },
        "meta": {"prediction": prediction},
    }


def aggregate_open_qa(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    del metric_options
    exact_match_values: list[float] = []
    token_f1_values: list[float] = []
    parsed_values: list[float] = []
    for item in sample_results:
        records = to_records(item["records"])
        if not records:
            continue
        exact_match_values.append(mean([float(record.score) for record in records]))
        token_f1_values.append(
            mean(
                [
                    float(record.parsed.get("token_f1", 0.0))
                    if isinstance(record.parsed, dict)
                    else 0.0
                    for record in records
                ]
            )
        )
        parsed_values.append(
            mean(
                [
                    float(
                        isinstance(record.parsed, dict)
                        and bool(record.parsed.get("prediction_normalized"))
                    )
                    for record in records
                ]
            )
        )
    return {
        "exact_match": mean(exact_match_values),
        "exact_match_stderr": mean_stderr(exact_match_values),
        "token_f1": mean(token_f1_values),
        "token_f1_stderr": mean_stderr(token_f1_values),
        "parsed_rate": mean(parsed_values),
    }
