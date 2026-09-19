import json
import re
import urllib.request
from pathlib import Path
from string import ascii_uppercase

# Official AGIEval release. HF mirrors such as dmayhem93/agieval-* lost option (D)
# of three SAT-English questions, one of which is the gold answer.
SOURCE_COMMIT = "84ab72d94318290aad2e4ec820d535a95a1f7552"
SOURCE_URL = (
    "https://raw.githubusercontent.com/ruixiangcui/AGIEval/"
    f"{SOURCE_COMMIT}/data/v1_1/{{subset}}.jsonl"
)


# The 8 tasks of OLMES `agi_eval_english` (sat-en-without-passage is not part of it).
ENGLISH_SUBSETS = [
    "aqua-rat",
    "gaokao-english",
    "logiqa-en",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "sat-en",
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


def _official_zero_shot_query(row: dict) -> str:
    # AGIEval src/dataset_loader.py convert_zero_shot for English QA datasets.
    passage = row["passage"] if row["passage"] is not None else ""
    options = row["options"]
    return (
        passage
        + "Q: "
        + row["question"]
        + " "
        + "Answer Choices: "
        + " ".join(options)
        + "\n"
        + f"A: Among A through {ascii_uppercase[len(options) - 1]}, the answer is"
    )


def _load_subset(subset: str) -> list[dict]:
    with urllib.request.urlopen(SOURCE_URL.format(subset=subset)) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for subset in ENGLISH_SUBSETS:
        source = SOURCE_URL.format(subset=subset)
        for idx, row in enumerate(_load_subset(subset)):
            query = _official_zero_shot_query(row).strip()
            question = _extract_question(query)
            choices_raw = [str(c) for c in row["options"]]
            choices_clean = [_normalize_choice(c) for c in choices_raw]
            if len(choices_clean) < 2 or any(not c for c in choices_clean):
                raise ValueError(f"Invalid choices for subset={subset} idx={idx}")

            label = str(row["label"]).strip()
            if len(label) != 1 or label not in ascii_uppercase[: len(choices_clean)]:
                raise ValueError(f"Invalid label for subset={subset} idx={idx}: {label!r}")

            choices = {
                ascii_uppercase[i]: choices_clean[i] for i in range(len(choices_clean))
            }
            answer = label

            rows.append(
                {
                    "id": f"{subset}_{idx:05d}",
                    "subset": subset,
                    "question": question,
                    "query": query,
                    "choices": choices,
                    "answer": answer,
                    "source": source,
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)} subsets={len(ENGLISH_SUBSETS)}")


if __name__ == "__main__":
    main()
