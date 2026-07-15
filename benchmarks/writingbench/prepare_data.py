#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/tmp/WritingBench/benchmark_query/benchmark_all.jsonl",
    )
    parser.add_argument(
        "--requirement-dir",
        default="/tmp/WritingBench/benchmark_query/requirement",
    )
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/eval.jsonl"))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    requirement_members: dict[str, set[int]] = {}
    requirement_criteria: dict[str, dict[int, list[str]]] = {}
    requirement_dir = Path(args.requirement_dir)
    for dimension in ("style", "format", "length"):
        subset = requirement_dir / dimension / f"{dimension}_subset.jsonl"
        subset_c = requirement_dir / dimension / f"{dimension}_subset_C.jsonl"
        requirement_members[dimension] = {
            int(json.loads(line)["index"]) for line in subset.open(encoding="utf-8")
        }
        requirement_criteria[dimension] = {
            int(row["index"]): [str(item["name"]) for item in row["checklist"]]
            for row in (json.loads(line) for line in subset_c.open(encoding="utf-8"))
        }

    with Path(args.source).open(encoding="utf-8") as src, output.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
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
