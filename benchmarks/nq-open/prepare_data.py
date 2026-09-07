"""Prepare the labeled public NQ-Open development split."""

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "natural-questions/master/nq_open/NQ-open.dev.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "data/eval.jsonl"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="aethereval-nq-") as temp:
        source = args.source or Path(temp) / "NQ-open.dev.jsonl"
        if not source.exists():
            urllib.request.urlretrieve(SOURCE_URL, source)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with (
            source.open(encoding="utf-8") as rows,
            args.output.open("w", encoding="utf-8") as destination,
        ):
            for index, line in enumerate(rows):
                row = json.loads(line)
                destination.write(
                    json.dumps(
                        {
                            "id": f"nq-open-dev-{index}",
                            "question": row["question"],
                            "answers": row["answer"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
    print(f"Wrote {count} NQ-Open dev examples to {args.output}")


if __name__ == "__main__":
    main()
