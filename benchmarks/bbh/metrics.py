
import re
import string
from typing import Any

from aethereval.core.types import GenerationRecord, Sample
from aethereval.metrics.common import aggregate_binary_results


PRIMARY_METRIC = "exact_match"


# OLMES BBH task-specific answer formats.
BBH_ANSWER_REGEX = {
    "boolean_expressions": "[tT]rue|[fF]alse",
    "causal_judgement": "[yY]es|[nN]o",
    "date_understanding": "MC",
    "disambiguation_qa": "MC",
    "dyck_languages": "[\\]\\)\\}\\> ]+",
    "formal_fallacies": "[iI]nvalid|[vV]alid",
    "geometric_shapes": "MC",
    "hyperbaton": "MC",
    "logical_deduction_five_objects": "MC",
    "logical_deduction_seven_objects": "MC",
    "logical_deduction_three_objects": "MC",
    "movie_recommendation": "MC",
    "multistep_arithmetic_two": "-?\\d+",
    "navigate": "[nN]o|[yY]es",
    "object_counting": "\\d+",
    "penguins_in_a_table": "MC",
    "reasoning_about_colored_objects": "MC",
    "ruin_names": "MC",
    "salient_translation_error_detection": "MC",
    "snarks": "MC",
    "sports_understanding": "[yY]es|[nN]o",
    "temporal_sequences": "MC",
    "tracking_shuffled_objects_five_objects": "MC",
    "tracking_shuffled_objects_seven_objects": "MC",
    "tracking_shuffled_objects_three_objects": "MC",
    "web_of_lies": "[yY]es|[nN]o",
    # Allow apostrophes / ampersands seen in BBH word_sorting targets.
    "word_sorting": "[a-z'&,-]+(?: [a-z'&,-]+)*",
}


_ANSWER_REGEX_TEMPLATES = [
    "(?i)So the answer is ($ANS$)\\.?",
    "(?i)answer is ($ANS$)",
    "(?i)answer:.*?($ANS$)",
    "(?i)answer\\b.*?($ANS$)",
    "($ANS$)",
]

_SPECIAL_DELIMITERS_TO_STRIP = [
    ("$", "$"),
    ("\\(", "\\)"),
    ("(", ")"),
    ("**", "**"),
    ("***", "***"),
    ("\\[", "\\]"),
    ("'", "'"),
    ("`", "`"),
    ('"', '"'),
]

_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def _extract_last(regex: str, text: str) -> str:
    found = re.findall(regex, text)
    if not found:
        return ""

    last = found[-1]
    if isinstance(last, tuple):
        for item in reversed(last):
            if item:
                return str(item)
        return ""
    return str(last)


def _extract_answer(generation: str, subset: str, gold: str) -> tuple[str, str]:
    answer_regex = BBH_ANSWER_REGEX.get(subset, "MC")
    is_mc = answer_regex == "MC"
    if is_mc:
        # OLMES default for MC tasks is parenthesized letter, but some BBH
        # rows contain free-form gold strings; fall back to exact gold text.
        if re.fullmatch(r"\([A-Z]\)", gold):
            answer_regex = "\\([A-Z]\\)"
        else:
            answer_regex = re.escape(gold)

    regexes = list(_ANSWER_REGEX_TEMPLATES)
    if is_mc:
        regexes.append("\\b([A-Z])\\b")
    regexes.append("(?i)($ANS$)")

    extracted = ""
    method = "none"
    for idx, template in enumerate(regexes):
        regex = template.replace("$ANS$", answer_regex)
        candidate = _extract_last(regex, generation)
        if candidate:
            extracted = candidate
            method = f"template_{idx}"
            break

    for left, right in _SPECIAL_DELIMITERS_TO_STRIP:
        if re.match(answer_regex, left):
            continue
        left_regex = re.escape(left)
        right_regex = re.escape(right)
        extracted = re.sub(f"^{left_regex}(.*){right_regex}$", "\\1", extracted).strip()

    if is_mc and len(extracted) == 1:
        extracted = f"({extracted})"

    return extracted, method


def _normalize_exact_match(text: str, *, ignore_punctuation: bool) -> str:
    normalized = str(text).strip().lower()
    if ignore_punctuation:
        normalized = normalized.translate(_PUNCT_TRANSLATION)
    return normalized


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    subset = str(sample.meta.get("subset", sample.data.get("subset", ""))).strip()
    gold = str(sample.gold).strip()
    prediction, method = _extract_answer(generation, subset, gold)

    ignore_punctuation = subset != "dyck_languages"
    prediction_norm = _normalize_exact_match(prediction, ignore_punctuation=ignore_punctuation)
    gold_norm = _normalize_exact_match(gold, ignore_punctuation=ignore_punctuation)

    score = 1.0 if prediction_norm == gold_norm and bool(gold_norm) else 0.0

    parsed = {
        "prediction": prediction,
        "prediction_norm": prediction_norm,
        "gold": gold,
        "gold_norm": gold_norm,
        "extract_method": method,
    }
    return {
        "score": score,
        "is_pass": bool(score),
        "parsed": parsed,
        "meta": {
            "subset": subset,
            "prediction": prediction,
            "extract_method": method,
        },
    }


def _record_has_prediction(record: GenerationRecord) -> bool:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    prediction_norm = parsed.get("prediction_norm")
    return isinstance(prediction_norm, str) and bool(prediction_norm)


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    result = aggregate_binary_results(
        sample_results,
        metric_options,
        parsed_flag_fn=_record_has_prediction,
        group_key="subset",
    )

    result["exact_match"] = float(result.get("accuracy", 0.0))
    result["exact_match_stderr"] = float(result.get("accuracy_stderr", 0.0))

    for key, value in list(result.items()):
        if key.startswith("accuracy_"):
            suffix = key[len("accuracy_") :]
            result[f"exact_match_{suffix}"] = value
        elif key.startswith("accuracy@"):
            suffix = key[len("accuracy") :]
            result[f"exact_match{suffix}"] = value

    return result
