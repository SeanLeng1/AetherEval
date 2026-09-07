#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="/tmp/LLMEval-Med")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "data"))
    args = parser.parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = json.loads((source / "dataset/dataset.json").read_text(encoding="utf-8"))
    with (output / "eval.jsonl").open("w", encoding="utf-8") as dst:
        for category, rows in dataset.items():
            for row in rows:
                payload = dict(row)
                payload["id"] = f"{category}-{row['groupCode']}-{row['round']}"
                payload["category"] = category
                dst.write(json.dumps(payload, ensure_ascii=False) + "\n")

    prompts = _extract_prompts(source / "evaluate/Evaluate.py")
    (output / "judge_prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _extract_prompts(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prompts: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("prompt_"):
            continue
        if not isinstance(node.value, ast.JoinedStr):
            continue
        pieces: list[str] = []
        for value in node.value.values:
            if isinstance(value, ast.Constant):
                pieces.append(str(value.value))
            elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                pieces.append(f"<<{value.value.id}>>")
            else:
                raise ValueError(f"Unsupported f-string component in {target.id}")
        prompts[target.id.removeprefix("prompt_")] = "".join(pieces)
    if len(prompts) != 5:
        raise ValueError(f"Expected five LLMEval-Med judge prompts, got {sorted(prompts)}")
    return prompts


if __name__ == "__main__":
    main()
