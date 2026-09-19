import re
import string
from typing import Any

from aethereval.core.types import GenerationRecord, Sample
from aethereval.metrics.common import aggregate_binary_results


PRIMARY_METRIC = "exact_match"


# OLMES BBH task-specific answer formats.
BBH_ANSWER_REGEX = {
    "boolean_expressions": "\\b(?:[tT]rue|[fF]alse)\\b",
    "causal_judgement": "\\b(?:[yY]es|[nN]o)\\b",
    "date_understanding": "MC",
    "disambiguation_qa": "MC",
    "dyck_languages": "[\\]\\)\\}\\> ]+",
    "formal_fallacies": "\\b(?:[iI]nvalid|[vV]alid)\\b",
    "geometric_shapes": "MC",
    "hyperbaton": "MC",
    "logical_deduction_five_objects": "MC",
    "logical_deduction_seven_objects": "MC",
    "logical_deduction_three_objects": "MC",
    "movie_recommendation": "MC",
    "multistep_arithmetic_two": "-?\\d[\\d,]*",
    "navigate": "\\b(?:[nN]o|[yY]es)\\b",
    "object_counting": "\\d[\\d,]*",
    "penguins_in_a_table": "MC",
    "reasoning_about_colored_objects": "MC",
    "ruin_names": "MC",
    "salient_translation_error_detection": "MC",
    "snarks": "MC",
    "sports_understanding": "\\b(?:[yY]es|[nN]o)\\b",
    "temporal_sequences": "MC",
    "tracking_shuffled_objects_five_objects": "MC",
    "tracking_shuffled_objects_seven_objects": "MC",
    "tracking_shuffled_objects_three_objects": "MC",
    "web_of_lies": "\\b(?:[yY]es|[nN]o)\\b",
    # Allow apostrophes / ampersands seen in BBH word_sorting targets.
    "word_sorting": "[a-z'&,-]+(?: [a-z'&,-]+)*",
}


# Tolerate "answer is: X" and markdown/quote markup before the answer; otherwise the
# lazy fallbacks below capture the word "is" for free-text subsets.
_ANSWER_LEAD = "[:\\s]*[*`\"'$]*\\s*"
_NUMERIC_SUBSETS = {"multistep_arithmetic_two", "object_counting"}

_ANSWER_REGEX_TEMPLATES = [
    "(?i)So the answer is" + _ANSWER_LEAD + "($ANS$)\\.?",
    "(?i)answer is" + _ANSWER_LEAD + "($ANS$)",
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


def _extract_answer(generation: str, subset: str) -> tuple[str, str]:
    answer_regex = BBH_ANSWER_REGEX.get(subset, "MC")
    is_mc = answer_regex == "MC"
    if is_mc:
        # Extract independently of the gold label, including malformed dataset rows.
        answer_regex = "\\([A-Z]\\)"

    regexes = list(_ANSWER_REGEX_TEMPLATES)
    if is_mc:
        # Zero-shot answers often give a bare letter ("the answer is B.", "\\boxed{B}").
        # Accept it at an explicit answer position, before the position-free fallbacks
        # that would otherwise pick up the pronoun "I" or the article "A".
        # Case-sensitive; "A"/"I" followed by a lowercase word is prose ("I think"),
        # unless that word starts a justification ("A because ...").
        bare = (
            "\\(?\\b(?-i:[B-HJ-Z]|[AI](?!\\s+(?!because\\b|since\\b)[a-z]))\\b\\)?"
        )
        anchored = [t.replace("$ANS$", bare) for t in _ANSWER_REGEX_TEMPLATES[:2]]
        anchored.append("(?i)answer:" + _ANSWER_LEAD + "(" + bare + ")")
        anchored.append("\\\\boxed\\{\\s*(?:\\\\text\\{)?\\(?([A-Z])\\)?")
        regexes = regexes[:2] + anchored + regexes[2:]
        # Last resort: a standalone capital anywhere, still skipping prose "I"/"A".
        regexes.append("(" + bare.replace("\\(?", "").replace("\\)?", "") + ")")
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

    letter = re.fullmatch("\\(?([A-Za-z])\\)?", extracted) if is_mc else None
    if letter:
        extracted = f"({letter.group(1)})"

    return extracted, method


def _normalize_exact_match(text: str, *, ignore_punctuation: bool) -> str:
    normalized = str(text).strip().lower()
    if ignore_punctuation:
        normalized = normalized.translate(_PUNCT_TRANSLATION)
    return normalized


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    subset = str(sample.meta.get("subset", sample.data.get("subset", ""))).strip()
    gold = str(sample.gold).strip()
    prediction, method = _extract_answer(generation, subset)

    if subset in _NUMERIC_SUBSETS:
        # Punctuation stripping would also delete the minus sign.
        prediction = prediction.replace(",", "").rstrip(".")
    ignore_punctuation = subset != "dyck_languages" and subset not in _NUMERIC_SUBSETS
    prediction_norm = _normalize_exact_match(
        prediction, ignore_punctuation=ignore_punctuation
    )
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
