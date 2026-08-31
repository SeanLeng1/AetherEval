"""Prepare TriviaQA's public unfiltered, no-context validation split."""

import argparse
import json
from pathlib import Path
from typing import Any


def _aliases(answer: dict[str, Any]) -> list[str]:
    values = [answer.get("value", ""), *list(answer.get("aliases") or [])]
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/eval.jsonl"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        "unfiltered.nocontext",
        split="validation",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for index, row in enumerate(dataset):
            destination.write(
                json.dumps(
                    {
                        "id": str(row.get("question_id") or index),
                        "question": str(row["question"]),
                        "answers": _aliases(dict(row["answer"])),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(dataset)} TriviaQA validation examples to {args.output}")


if __name__ == "__main__":
    main()
