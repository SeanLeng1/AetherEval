#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path

from aethereval.core.io import write_jsonl


SOURCE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_URL)
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/eval.jsonl"))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if str(args.source).startswith(("http://", "https://")):
        source = urllib.request.urlopen(args.source)
    else:
        source = Path(args.source).open("rb")
    rows: list[dict] = []
    with source:
        for idx, raw in enumerate(source):
            row = json.loads(raw)
            rows.append(
                {
                    "id": str(row.get("prompt_id", idx)),
                    "prompt_id": row.get("prompt_id", idx),
                    "prompt": row["prompt"],
                    "rubrics": row["rubrics"],
                    "example_tags": row.get("example_tags", []),
                }
            )
    write_jsonl(output, rows)


if __name__ == "__main__":
    main()
