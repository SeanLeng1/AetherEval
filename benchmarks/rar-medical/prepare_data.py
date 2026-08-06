#!/usr/bin/env python3

import argparse
from pathlib import Path

from benchmark_utils.rar_data import DATASETS, prepare_rar_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASETS["Medical"])
    parser.add_argument(
        "--output", default=str(Path(__file__).parent / "data/eval.jsonl")
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    prepare_rar_data(
        domain="Medical",
        output=args.output,
        dataset_id=args.dataset,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
