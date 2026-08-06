"""Independent ScaleAI Rubrics-as-Rewards data preparation for AetherEval."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from aethereval.core.io import write_jsonl
from benchmark_utils.rar_protocol import build_grader_prompt


DATASETS = {
    "Medical": "ScaleAI/RaR-Medicine",
    "Science": "ScaleAI/RaR-Science",
}
DEFAULT_ACTOR_TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GRADER_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_MAX_PROMPT_LENGTH = 1024
DEFAULT_GRADER_MAX_MODEL_LEN = 32768
DEFAULT_GRADER_MAX_TOKENS = 4096
DEFAULT_RESPONSE_TOKEN_RESERVE = 4096
FAMILY_POINTS = {
    "Essential": 1.0,
    "Pitfall": 0.9,
    "Important": 0.7,
    "Optional": 0.3,
}
FAMILY_NAMES = tuple(FAMILY_POINTS)

_KNOWN_BAD_QUESTIONS = {
    "Medical": {
        "A patient with history of HTN treated with captopril came to office "
        "with angioneurotic edema. What would be the cause?"
    },
    "Science": set(),
}
_CATEGORY_RE = re.compile(
    r"^\s*(essential|esssential|important|importance|optional|option|pitfall|"
    r"mandatory|universal)(?:\s+criteria)?(?:\s*:|\s+that\b)\s*",
    re.IGNORECASE,
)
_TITLE_CATEGORY_RE = re.compile(
    r"^\s*(essential|esssential|important|importance|optional|option|pitfall|"
    r"mandatory|universal)(?:\s+criteria)?\s*$",
    re.IGNORECASE,
)
_CATEGORY_ALIASES = {
    "essential": "Essential",
    "esssential": "Essential",
    "mandatory": "Essential",
    "universal": "Essential",
    "important": "Important",
    "importance": "Important",
    "optional": "Optional",
    "option": "Optional",
    "pitfall": "Pitfall",
}
_MEDICAL_WEIGHT_FALLBACK = {
    5: "Essential",
    4: "Important",
    3: "Important",
    2: "Optional",
    1: "Optional",
    -1: "Pitfall",
    -2: "Pitfall",
}
_VALID_WEIGHTS = frozenset(_MEDICAL_WEIGHT_FALLBACK)


def parse_domain(value: str) -> str:
    lookup = {domain.lower(): domain for domain in DATASETS}
    key = str(value).strip().lower()
    if key not in lookup:
        raise ValueError(f"unknown RaR domain {value!r}; choose Medical or Science")
    return lookup[key]


def clean_text(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} must be nonempty")
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError(f"{field} must be nonempty")
    return text


def canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def _match_category(text: str, pattern: re.Pattern[str]) -> tuple[str, int] | None:
    match = pattern.match(text)
    if match is None:
        return None
    return _CATEGORY_ALIASES[match.group(1).lower()], match.end()


def rubric_family(raw: Mapping[str, Any], *, domain: str) -> tuple[str, str]:
    """Return a canonical family and criterion, repairing source label typos."""

    domain = parse_domain(domain)
    description = clean_text(raw.get("description"), field="rubric.description")
    title = clean_text(raw.get("title"), field="rubric.title")
    try:
        weight = int(raw["weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid rubric weight for {title!r}") from exc
    if weight not in _VALID_WEIGHTS:
        raise ValueError(f"unsupported RaR rubric weight {weight} for {title!r}")

    matched = _match_category(description, _CATEGORY_RE)
    if matched is not None:
        family, body_start = matched
        body = description[body_start:].strip()
    else:
        title_match = _match_category(title, _TITLE_CATEGORY_RE)
        if title_match is not None:
            family, body = title_match[0], description
        elif domain == "Medical":
            family, body = _MEDICAL_WEIGHT_FALLBACK[weight], description
        elif weight < 0:
            family, body = "Pitfall", description
        else:
            raise ValueError(
                f"cannot infer RaR-{domain} category for rubric {title!r}: "
                f"{description!r}"
            )

    if not body:
        raise ValueError(f"rubric {title!r} has an empty criterion body")
    if (family == "Pitfall") != (weight < 0):
        raise ValueError(
            f"rubric category/weight sign mismatch for {title!r}: "
            f"family={family}, weight={weight}"
        )
    return family, body


def normalize_rubrics(
    example: Mapping[str, Any], *, domain: str
) -> list[dict[str, Any]]:
    raw_rubrics = example.get("rubric")
    if not isinstance(raw_rubrics, Sequence) or isinstance(raw_rubrics, (str, bytes)):
        raise TypeError("RaR rubric must be a sequence")
    normalized: list[dict[str, Any]] = []
    for raw in raw_rubrics:
        if not isinstance(raw, Mapping):
            raise TypeError("RaR rubric entries must be objects")
        family, criterion = rubric_family(raw, domain=domain)
        normalized.append(
            {
                "criterion": criterion,
                "points": FAMILY_POINTS[family],
                "family": family,
                "raw_weight": int(raw["weight"]),
                "title": clean_text(raw.get("title"), field="rubric.title"),
            }
        )
    if not normalized:
        raise ValueError("RaR sample must contain at least one rubric")
    return normalized


def validate_source_row(example: Mapping[str, Any], *, domain: str) -> None:
    clean_text(example.get("question"), field="question")
    clean_text(example.get("reference_answer"), field="reference_answer")
    clean_text(example.get("question_source"), field="question_source")
    rubrics = example.get("rubric")
    rubric_list = example.get("rubric_list")
    if not isinstance(rubrics, Sequence) or isinstance(rubrics, (str, bytes)):
        raise TypeError("RaR rubric must be a sequence")
    if not isinstance(rubric_list, Sequence) or isinstance(rubric_list, (str, bytes)):
        raise TypeError("RaR rubric_list must be a sequence")
    if int(example.get("rubric_count", -1)) != len(rubrics):
        raise ValueError("rubric_count disagrees with rubric length")
    if [str(item.get("description", "")) for item in rubrics] != [
        str(value) for value in rubric_list
    ]:
        raise ValueError("rubric_list disagrees with rubric descriptions")
    normalize_rubrics(example, domain=domain)


def clean_source_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    domain: str,
    split: str,
    blocked_questions: set[str] | None = None,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Validate, decontaminate, and deterministically deduplicate a split."""

    domain = parse_domain(domain)
    blocked = blocked_questions or set()
    known_bad = {
        canonical_text(question) for question in _KNOWN_BAD_QUESTIONS[domain]
    }
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    rejected_known = rejected_overlap = rejected_malformed = 0
    source_count = 0
    for index, row in enumerate(rows):
        source_count += 1
        try:
            validate_source_row(row, domain=domain)
        except (KeyError, TypeError, ValueError):
            rejected_malformed += 1
            continue
        key = canonical_text(str(row["question"]))
        if key in known_bad:
            rejected_known += 1
            continue
        if key in blocked:
            rejected_overlap += 1
            continue
        groups[key].append((index, row))

    kept: list[tuple[int, Mapping[str, Any]]] = []
    duplicate_excess = conflicting_groups = 0
    for candidates in groups.values():
        if len(candidates) == 1:
            kept.append(candidates[0])
            continue
        duplicate_excess += len(candidates) - 1
        references = {
            canonical_text(str(row["reference_answer"])) for _, row in candidates
        }
        if len(references) != 1:
            conflicting_groups += 1
            continue
        kept.append(
            max(
                candidates,
                key=lambda item: (int(item[1]["rubric_count"]), -item[0]),
            )
        )
    kept.sort(key=lambda item: item[0])
    print(
        f"RaR-{domain} {split} cleaning: kept {len(kept)}/{source_count}; "
        f"known_bad={rejected_known}, cross_split_overlap={rejected_overlap}, "
        f"malformed={rejected_malformed}, duplicate_excess={duplicate_excess}, "
        f"conflicting_duplicate_groups={conflicting_groups}"
    )
    return kept


def transform_eval_row(
    row: Mapping[str, Any], source_index: int, *, domain: str, dataset_id: str
) -> dict[str, Any]:
    domain = parse_domain(domain)
    question = clean_text(row.get("question"), field="question")
    return {
        "id": f"rar-{domain.lower()}-{source_index:06d}",
        "prompt": [{"role": "user", "content": question}],
        "reference_answer": clean_text(
            row.get("reference_answer"), field="reference_answer"
        ),
        "rubrics": normalize_rubrics(row, domain=domain),
        "meta": {
            "domain": domain,
            "question_source": clean_text(
                row.get("question_source"), field="question_source"
            ),
            "rubric_count": int(row["rubric_count"]),
            "source_index": int(source_index),
            "source": dataset_id,
            "split": "test",
        },
    }


def _token_lengths(tokenizer: Any, rendered: list[str], *, batch_size: int) -> list[int]:
    lengths: list[int] = []
    for offset in range(0, len(rendered), batch_size):
        encoded = tokenizer(
            rendered[offset : offset + batch_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        lengths.extend(len(tokens) for tokens in encoded)
    return lengths


def filter_eval_contexts(
    records: list[dict[str, Any]],
    *,
    actor_tokenizer: Any,
    grader_tokenizer: Any,
    max_prompt_length: int,
    grader_max_model_len: int,
    grader_max_tokens: int,
    response_token_reserve: int,
) -> list[dict[str, Any]]:
    """Match AetherRL's actor and all-rubric grader context filters."""

    limits = (
        int(max_prompt_length),
        int(grader_max_model_len),
        int(grader_max_tokens),
        int(response_token_reserve),
    )
    if any(value < 1 for value in limits):
        raise ValueError("RaR context limits must be positive")
    reserved = int(grader_max_tokens) + int(response_token_reserve)
    if reserved >= int(grader_max_model_len):
        raise ValueError(
            "grader output and response reserves must leave room for its prompt"
        )

    actor_rendered = [
        actor_tokenizer.apply_chat_template(
            record["prompt"], tokenize=False, add_generation_prompt=True
        )
        for record in records
    ]
    actor_lengths = _token_lengths(actor_tokenizer, actor_rendered, batch_size=256)
    actor_kept = [
        record
        for record, length in zip(records, actor_lengths, strict=True)
        if length <= int(max_prompt_length)
    ]
    print(
        f"actor prompt length: kept {len(actor_kept)}/{len(records)}, filtered "
        f"{len(records) - len(actor_kept)} above {int(max_prompt_length)} tokens"
    )

    grader_rendered = [
        grader_tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": build_grader_prompt(
                        record["prompt"], "", record["rubrics"]
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for record in actor_kept
    ]
    grader_lengths = _token_lengths(grader_tokenizer, grader_rendered, batch_size=128)
    grader_budget = int(grader_max_model_len) - reserved - 1
    grader_kept = [
        record
        for record, length in zip(actor_kept, grader_lengths, strict=True)
        if length <= grader_budget
    ]
    print(
        f"grader context: kept {len(grader_kept)}/{len(actor_kept)}, filtered "
        f"{len(actor_kept) - len(grader_kept)} above "
        f"{int(grader_max_model_len)} tokens"
    )
    return grader_kept


def prepare_rar_data(
    *,
    domain: str,
    output: str | Path,
    dataset_id: str | None = None,
    max_samples: int | None = None,
    actor_tokenizer: str = DEFAULT_ACTOR_TOKENIZER,
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH,
    grader_model: str = DEFAULT_GRADER_MODEL,
    grader_max_model_len: int = DEFAULT_GRADER_MAX_MODEL_LEN,
    grader_max_tokens: int = DEFAULT_GRADER_MAX_TOKENS,
    response_token_reserve: int = DEFAULT_RESPONSE_TOKEN_RESERVE,
) -> int:
    """Download the official test split and write AetherEval ``eval.jsonl``."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by the CLI user
        raise RuntimeError(
            "datasets and transformers are required for RaR preparation; install "
            "them with `pip install datasets transformers`."
        ) from exc

    domain = parse_domain(domain)
    source = dataset_id or DATASETS[domain]
    raw_test = load_dataset(source, split="test")
    clean_test = clean_source_rows(
        raw_test,
        domain=domain,
        split="test",
    )
    if max_samples is not None:
        if int(max_samples) < 1:
            raise ValueError("max_samples must be positive")
    records = [
        transform_eval_row(row, index, domain=domain, dataset_id=source)
        for index, row in clean_test
    ]
    records = filter_eval_contexts(
        records,
        actor_tokenizer=AutoTokenizer.from_pretrained(actor_tokenizer),
        grader_tokenizer=AutoTokenizer.from_pretrained(grader_model),
        max_prompt_length=max_prompt_length,
        grader_max_model_len=grader_max_model_len,
        grader_max_tokens=grader_max_tokens,
        response_token_reserve=response_token_reserve,
    )
    if max_samples is not None:
        records = records[: int(max_samples)]
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    print(f"wrote {output_path} rows={len(records)} source={source}:test")
    return len(records)


__all__ = [
    "DATASETS",
    "DEFAULT_ACTOR_TOKENIZER",
    "DEFAULT_GRADER_MAX_MODEL_LEN",
    "DEFAULT_GRADER_MAX_TOKENS",
    "DEFAULT_GRADER_MODEL",
    "DEFAULT_MAX_PROMPT_LENGTH",
    "DEFAULT_RESPONSE_TOKEN_RESERVE",
    "FAMILY_NAMES",
    "FAMILY_POINTS",
    "canonical_text",
    "clean_source_rows",
    "filter_eval_contexts",
    "normalize_rubrics",
    "parse_domain",
    "prepare_rar_data",
    "rubric_family",
    "transform_eval_row",
]
