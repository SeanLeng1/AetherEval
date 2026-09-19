#!/usr/bin/env python3
import argparse
import json
from io import StringIO
from pathlib import Path

from benchmark_utils.data import read_text


SOURCE_ROOT = (
    "https://raw.githubusercontent.com/X-PLUG/WritingBench/"
    "ae2d5176449b7b769815482641d35926f26793eb/benchmark_query"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=f"{SOURCE_ROOT}/benchmark_all.jsonl",
    )
    parser.add_argument(
        "--requirement-dir",
        default=f"{SOURCE_ROOT}/requirement",
    )
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/eval.jsonl"))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    requirement_members: dict[str, set[int]] = {}
    requirement_criteria: dict[str, dict[int, list[str]]] = {}
    requirement_dir = args.requirement_dir.rstrip("/")
    for dimension in ("style", "format", "length"):
        subset = f"{requirement_dir}/{dimension}/{dimension}_subset.jsonl"
        subset_c = f"{requirement_dir}/{dimension}/{dimension}_subset_C.jsonl"
        requirement_members[dimension] = {
            int(json.loads(line)["index"]) for line in StringIO(read_text(subset))
        }
        requirement_criteria[dimension] = {
            int(row["index"]): [str(item["name"]) for item in row["checklist"]]
            for row in (json.loads(line) for line in StringIO(read_text(subset_c)))
        }

    source = StringIO(read_text(args.source))
    with output.open("w", encoding="utf-8") as dst:
        for line in source:
            row = json.loads(line)
            index = int(row["index"])
            row["requirement_subsets"] = [
                dimension
                for dimension in ("style", "format", "length")
                if index in requirement_members[dimension]
            ]
            row["requirement_criteria"] = {
                dimension: requirement_criteria[dimension].get(index, [])
                for dimension in ("style", "format", "length")
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
