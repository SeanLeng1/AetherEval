#!/usr/bin/env python3
import argparse
import json
import urllib.request
from pathlib import Path


SOURCE_URL = "https://huggingface.co/datasets/realliyifei/ResearchQA/resolve/main/test.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_URL)
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/eval.jsonl"))
    args = parser.parse_args()
    if str(args.source).startswith(("http://", "https://")):
        with urllib.request.urlopen(args.source) as response:
            rows = json.load(response)
    else:
        rows = json.loads(Path(args.source).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
