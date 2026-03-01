
import json
import re
from pathlib import Path
from string import ascii_uppercase


ENGLISH_SUBSETS = [
    "aqua-rat",
    "gaokao-english",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
    "sat-en-without-passage",
    "sat-math",
]

_CHOICE_PREFIX_RE = re.compile(r"^\([A-Z]\)\s*")


def _extract_question(query: str) -> str:
    text = query.strip()
    if text.startswith("Q:"):
        text = text[2:].strip()
    if "Answer Choices:" in text:
        text = text.split("Answer Choices:", 1)[0].strip()
    return text if text else query.strip()


def _normalize_choice(choice: str) -> str:
    text = str(choice).strip()
    cleaned = _CHOICE_PREFIX_RE.sub("", text).strip()
    return cleaned if cleaned else text


def _gold_index(raw_gold: object, num_choices: int) -> int:
    if isinstance(raw_gold, list):
        if not raw_gold:
            raise ValueError("Empty gold list")
        value = int(raw_gold[0])
    else:
        value = int(raw_gold)

    # Some AGIEval rows have 3 choices but gold=3; interpret this edge case as 1-based.
    if value == num_choices and num_choices >= 2:
        return num_choices - 1
    return value


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for prepare_data.py. Install with `pip install datasets`."
        ) from exc

    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for subset in ENGLISH_SUBSETS:
        repo = f"dmayhem93/agieval-{subset}"
        ds = load_dataset(repo, "default", split="test")

        for idx, row in enumerate(ds):
            query = str(row["query"]).strip()
            question = _extract_question(query)
            choices_raw = [str(c) for c in row["choices"]]
            choices_clean = [_normalize_choice(c) for c in choices_raw]
            if len(choices_clean) < 2 or any(not c for c in choices_clean):
                raise ValueError(f"Invalid choices for subset={subset} idx={idx}")

            gold_idx = _gold_index(row["gold"], len(choices_clean))
            if gold_idx < 0 or gold_idx >= len(choices_clean):
                raise ValueError(f"Gold index out of range for subset={subset} idx={idx}")

            choices = {
                ascii_uppercase[i]: choices_clean[i]
                for i in range(len(choices_clean))
            }
            answer = ascii_uppercase[gold_idx]

            rows.append(
                {
                    "id": f"{subset}_{idx:05d}",
                    "subset": subset,
                    "question": question,
                    "query": query,
                    "choices": choices,
                    "answer": answer,
                    "source": repo,
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)} subsets={len(ENGLISH_SUBSETS)}")


if __name__ == "__main__":
    main()
