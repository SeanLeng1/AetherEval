from __future__ import annotations

import json
from pathlib import Path

import requests


INPUT_DATA_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(INPUT_DATA_URL, timeout=60)
    resp.raise_for_status()

    kept_lines: list[str] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Validate each line is proper JSON.
        json.loads(line)
        kept_lines.append(line)

    with out_path.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    print(f"wrote {out_path} rows={len(kept_lines)}")


if __name__ == "__main__":
    main()
