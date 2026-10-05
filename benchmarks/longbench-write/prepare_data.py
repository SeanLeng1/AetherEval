import json
from pathlib import Path

from benchmark_utils.data import read_text, write_task_jsonl

REVISION = "447539b356a8b09760b51eca876e19b6fc1f2dd7"
SOURCE_URL = f"https://raw.githubusercontent.com/THUDM/LongWriter/{REVISION}/evaluation/longbench_write.jsonl"
ENGLISH_SOURCE_URL = f"https://raw.githubusercontent.com/THUDM/LongWriter/{REVISION}/evaluation/longbench_write_en.jsonl"


def main():
    rows = [
        json.loads(line) for line in read_text(SOURCE_URL).splitlines() if line.strip()
    ]
    if len(rows) != 120:
        raise ValueError("Expected the full 120-prompt LongBench-Write release")
    english = [
        json.loads(line)
        for line in read_text(ENGLISH_SOURCE_URL).splitlines()
        if line.strip()
    ]
    english_prompts = {row["prompt"] for row in english}
    if len(english_prompts) != 60 or not english_prompts.issubset(
        {row["prompt"] for row in rows}
    ):
        raise ValueError("English language labels do not match the official subset")
    rows = [
        {**row, "language": "en" if row["prompt"] in english_prompts else "zh"}
        for row in rows
    ]
    write_task_jsonl(Path(__file__).parent, rows)


if __name__ == "__main__":
    main()
