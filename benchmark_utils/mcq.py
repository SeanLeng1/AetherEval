import re
from functools import lru_cache
from pathlib import Path
from string import ascii_uppercase
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


def load_mcq_samples(
    task_dir: Path,
    data_file: str,
    benchmark_name: str,
    *,
    meta_keys: tuple[str, ...],
    raw_meta_keys: tuple[str, ...] = (),
    data_keys: tuple[str, ...] = (),
    letters: tuple[str, ...] | None = None,
) -> list[Sample]:
    """Load rows with an id, a question, an answer letter and a choices object.

    ``meta_keys`` and ``data_keys`` are stored as stripped strings and
    ``raw_meta_keys`` unchanged. With ``letters`` every listed choice is required;
    otherwise choices run A, B, ... up to the first missing label.
    """
    samples: list[Sample] = []
    for row in read_jsonl(task_dir / data_file):
        if not isinstance(row, dict):
            raise ValueError(f"{benchmark_name} row must be a JSON object")

        sample_id = str(row["id"])
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip().upper()
        if not question:
            raise ValueError(f"Empty question for sample {sample_id}")

        raw_choices = row["choices"]
        if not isinstance(raw_choices, dict):
            raise ValueError(f"choices must be a JSON object for sample {sample_id}")
        choices: dict[str, str] = {}
        for letter in letters or ascii_uppercase:
            if letters is None and letter not in raw_choices:
                break
            value = str(raw_choices[letter]).strip()
            if not value:
                raise ValueError(f"Missing choice '{letter}'")
            choices[letter] = value
        if len(choices) < 2:
            raise ValueError(f"{benchmark_name} item must include at least 2 choices")

        if answer not in choices:
            raise ValueError(f"Invalid answer label for sample {sample_id}: {answer}")

        meta: dict[str, Any] = {key: str(row.get(key, "")).strip() for key in meta_keys}
        meta.update((key, row.get(key)) for key in raw_meta_keys)
        samples.append(
            Sample(
                id=sample_id,
                gold=answer,
                meta=meta,
                data={
                    "question": question,
                    "choices": choices,
                    **{key: str(row.get(key, "")).strip() for key in data_keys},
                },
            )
        )
    return samples


_MCQ_PARSE_TAIL_CHARS = 1_000


def _prepare_mcq_parse_text(text: str) -> str:
    if len(text) <= _MCQ_PARSE_TAIL_CHARS:
        return text
    return text[-_MCQ_PARSE_TAIL_CHARS:]


def _normalize_choice(value: str, valid_set: set[str]) -> str | None:
    text = value.strip().upper()
    text = text.replace("(", "").replace(")", "")
    text = text.replace(".", "").replace(":", "")
    if not text:
        return None
    head = text[0]
    return head if head in valid_set else None


@lru_cache(maxsize=32)
def _choice_patterns(valid_letters: str) -> list[tuple[str, re.Pattern[str]]]:
    char_class = "".join(re.escape(letter) for letter in valid_letters)
    # Match standalone choice letters only (avoid picking letters inside words like "because").
    # A bare letter must be upper case so the article "a" is never read as choice A;
    # a parenthesised letter may be either case.
    choice_re = (
        rf"(?<![A-Za-z0-9])(?:\([{char_class}]\)|(?-i:[{char_class}])\)?)(?![A-Za-z0-9])"
    )
    # At an explicit answer position, also allow a lowercase letter terminated by
    # punctuation or the end of the line, but not the article in "a catalyst".
    answer_re = (
        rf"(?:{choice_re}|(?<![A-Za-z0-9])[{char_class}]"
        rf"(?=[ \t]*(?:[.)\],:;!?]|\r?$)))"
    )
    boxed_re = rf"\\boxed\{{\s*(?P<choice>[{char_class}])\s*\}}"
    return [
        # A terminal box is an explicit final answer, even when an earlier
        # sentence named another option. Allow trailing LaTeX/Markdown delimiters.
        (
            "boxed_final",
            re.compile(boxed_re + r"[\s.$\\\[\]{}*]*\Z", re.IGNORECASE),
        ),
        (
            "final_answer",
            re.compile(
                rf"(?im)final\s+answer(?:\s+is)?\s*[:：]?\s*(?P<choice>{answer_re})"
            ),
        ),
        (
            "answer_colon",
            re.compile(rf"(?im)\banswer\s*[:：]\s*(?P<choice>{answer_re})"),
        ),
        # Non-terminal boxes must not override a later explicit final answer.
        ("boxed", re.compile(boxed_re, re.IGNORECASE)),
        (
            "answer_anchor",
            re.compile(rf"(?i)\banswer\b.{{0,80}}?(?P<choice>{choice_re})"),
        ),
        (
            "option_anchor",
            re.compile(
                rf"(?i)\b(?:option|choice)\b(?:\s+is)?\s*(?P<choice>{choice_re})"
            ),
        ),
        (
            "select_anchor",
            re.compile(
                rf"(?i)\b(?:choose|chosen|select|selected|pick|picked)\b.{{0,40}}?(?P<choice>{choice_re})"
            ),
        ),
        # The leading run excludes newlines: "^\s*\s*" backtracks cubically on
        # whitespace-only tails. The later "\s*" still spans lines, so matches are unchanged.
        (
            "line_start",
            re.compile(
                rf"(?im)^[^\S\n]*(?:\*\*)?\s*(?P<choice>\(?[{char_class}]\)?)(?:\*\*)?\s*(?:[\)\].,:]|$)"
            ),
        ),
    ]


def extract_choice(
    text: str,
    choices: dict[str, str],
    valid_letters: list[str],
) -> tuple[str | None, str]:
    del choices
    valid_set = set(valid_letters)
    patterns = _choice_patterns("".join(valid_letters))
    parse_text = _prepare_mcq_parse_text(text)

    best: tuple[int, int, str, str] | None = None
    for priority, (method, pattern) in enumerate(patterns):
        for match in pattern.finditer(parse_text):
            choice = _normalize_choice(match.group("choice"), valid_set)
            if choice is None:
                continue
            candidate = (priority, -match.end(), choice, method)
            if best is None or candidate < best:
                best = candidate

    if best is not None:
        _, _, choice, method = best
        return choice, method

    return None, "none"


def score_generation_mcq(sample: Sample, generation: str) -> dict[str, Any]:
    choices = sample.data["choices"]
    if not isinstance(choices, dict) or not choices:
        raise ValueError(f"choices must be a non-empty dict for sample {sample.id}")

    valid_letters = [
        k.strip().upper() for k in sorted(choices.keys()) if str(k).strip()
    ]
    if not valid_letters:
        raise ValueError(
            f"choices must include non-empty labels for sample {sample.id}"
        )

    prediction, method = extract_choice(generation, choices, valid_letters)
    gold = str(sample.gold).strip().upper()
    score = 1.0 if prediction == gold else 0.0

    parsed = {
        "prediction": prediction,
        "gold": gold,
        "extract_method": method,
    }
    return {
        "score": score,
        "is_pass": bool(score),
        "parsed": parsed,
        "meta": {
            "prediction": prediction,
            "extract_method": method,
        },
    }
