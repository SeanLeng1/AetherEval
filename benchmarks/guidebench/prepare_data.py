"""Prepare the six categories actually published in the official repository."""

import json
from pathlib import Path

from benchmark_utils.data import read_text, write_task_jsonl

REVISION = "78c5bfa42facee34db31e4ba03ad2c3b5a04bbbc"
CATEGORY_COUNTS = {
    "price": 442,
    "relevance": 192,
    "math": 52,
    "chat": 180,
    "summary": 58,
    "hallu": 118,
}


def main():
    rows = []
    for category, expected in CATEGORY_COUNTS.items():
        source = f"https://raw.githubusercontent.com/Dlxxx/GuideBench/{REVISION}/data/{category}_tasks.json"
        data = json.loads(read_text(source))
        if not isinstance(data, list) or len(data) != expected:
            raise ValueError(f"GuideBench {category}: expected {expected} records")
        rows.extend(
            {**row, "category": category, "id": f"{category}:{index:04d}"}
            for index, row in enumerate(data)
        )
    write_task_jsonl(
        Path(__file__).parent, rows, " official public six-category release"
    )


if __name__ == "__main__":
    main()
