#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from aethereval.core.io import write_jsonl
from benchmark_utils.data import read_text


SOURCE_URL = (
    "https://huggingface.co/datasets/realliyifei/ResearchQA/resolve/"
    "bf8a4cfef073ecfc0275c57acf8ca960e4dc79d6/test.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_URL)
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/eval.jsonl"))
    args = parser.parse_args()
    rows = json.loads(read_text(args.source))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)


if __name__ == "__main__":
    main()
