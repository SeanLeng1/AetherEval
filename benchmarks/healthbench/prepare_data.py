#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path


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
    with source, output.open("w", encoding="utf-8") as dst:
        for idx, raw in enumerate(source):
            row = json.loads(raw)
            payload = {
                "id": str(row.get("prompt_id", idx)),
                "prompt_id": row.get("prompt_id", idx),
                "prompt": row["prompt"],
                "rubrics": row["rubrics"],
                "example_tags": row.get("example_tags", []),
            }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
