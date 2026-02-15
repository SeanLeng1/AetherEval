from __future__ import annotations

import re
from typing import Any

from aethereval.core.types import Sample
from aethereval.metrics.common import aggregate_mcq_results


PRIMARY_METRIC = "accuracy"

_OLMO_3_REGEXES = [
    r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?($ANS$)\b",
    r"(?i)so\s+the\s+answer\s+is\s+($ANS$)\.?",
    r"(?i)the\s+correct\s+answer\s+is:?\s*($ANS$)",
    r"(?i)the\s+answer\s+is\s+($ANS$)\.?",
    r"(?i)a:\s*($ANS$)",
    r"(?i)answer:\s*($ANS$)",
    r"(?i)\b($ANS$)\)?\s+is\s+correct",
    r"(?i)\(($ANS$)\)",
    r"(?i)\b($ANS$)\b",
    r"(?i)\b($ANS$)\b",
    r"(?i).*\b($ANS$)\b",
]


def _normalize_prediction(raw: str, valid_letters: set[str]) -> str | None:
    text = str(raw).strip().upper().replace("(", "").replace(")", "")
    if not text:
        return None
    head = text[0]
    return head if head in valid_letters else None


def _extract_olmo3_style_answer(generation: str, letters: list[str]) -> tuple[str | None, str, float]:
    if not letters:
        return None, "none", 0.0

    class_part = "".join(re.escape(letter) for letter in letters)
    answer_format_regex = rf"Therefore, the answer is \(([{class_part}])\)"
    answer_regex = rf"\(?([{class_part}])\)?"

    answer_string = ""
    method = "none"
    answer_format_correct = 0.0

    # Match the exact target format first.
    exact_matches = re.findall(answer_format_regex, generation)
    if exact_matches:
        answer_string = exact_matches[-1]
        answer_format_correct = 1.0
        method = "answer_format_regex"

    # Then try OLMo-3 regex templates in order.
    if answer_string == "":
        for idx, template in enumerate(_OLMO_3_REGEXES):
            regex = template.replace("$ANS$", answer_regex)
            matches = list(re.finditer(regex, generation))
            if not matches:
                continue
            match = matches[-1]
            groups = match.groups()
            if groups:
                answer_string = next((g for g in reversed(groups) if g), groups[0])
            else:
                answer_string = match.group(0)
            if answer_string != "":
                answer_format_correct = 1.0 if idx == 0 else 0.5
                method = f"template_{idx}"
                break

    # Final raw fallback on answer regex.
    if answer_string == "":
        raw_matches = re.findall(answer_regex, generation)
        if raw_matches:
            answer_string = raw_matches[-1]
            answer_format_correct = 0.2
            method = "raw_answer_regex"

    pred = _normalize_prediction(answer_string, set(letters))
    return pred, method, answer_format_correct


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    choices = sample.data.get("choices", {})
    if not isinstance(choices, dict):
        choices = {}
    letters = [str(k).strip().upper() for k in sorted(choices.keys()) if str(k).strip()]
    if not letters:
        letters = ["A", "B", "C", "D", "E"]

    prediction, method, answer_format_correct = _extract_olmo3_style_answer(generation, letters)
    gold = str(sample.gold).strip().upper()
    score = 1.0 if prediction == gold else 0.0

    parsed = {
        "prediction": prediction,
        "gold": gold,
        "extract_method": method,
        "answer_format_correct": answer_format_correct,
    }
    return {
        "score": score,
        "is_pass": bool(score),
        "parsed": parsed,
        "meta": {
            "prediction": prediction,
            "extract_method": method,
            "answer_format_correct": answer_format_correct,
        },
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    return aggregate_mcq_results(sample_results, metric_options, group_key="subset")
