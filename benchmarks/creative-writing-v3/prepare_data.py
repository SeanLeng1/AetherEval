#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/tmp/creative-writing-bench/data/creative_writing_prompts_v3.json",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default=str(Path(__file__).parent / "data/eval.jsonl"))
    args = parser.parse_args()
    prompts = json.loads(Path(args.source).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dst:
        for prompt_id, item in prompts.items():
            seeds = item["seed_modifiers"]
            for iteration in range(1, args.iterations + 1):
                payload = {
                    "id": f"{prompt_id}-{iteration}",
                    "prompt_id": str(prompt_id),
                    "iteration": iteration,
                    "category": item.get("category", ""),
                    "title": item.get("title", ""),
                    "base_prompt": item["writing_prompt"],
                    "seed_modifier": seeds[(iteration - 1) % len(seeds)],
                }
                dst.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
