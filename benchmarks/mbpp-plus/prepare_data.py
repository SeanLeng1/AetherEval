"""Download a pinned raw MBPP+ release once; evaluation then needs no network."""

import gzip
import json
from pathlib import Path
from urllib.request import urlopen

from benchmark_utils.data import write_text


VERSION = "v0.2.0"
URL = (
    "https://github.com/evalplus/mbppplus_release/releases/download/"
    f"{VERSION}/MbppPlus.jsonl.gz"
)


def main() -> None:
    with urlopen(URL, timeout=120) as response:
        text = gzip.decompress(response.read()).decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) != 378 or len({row["task_id"] for row in rows}) != 378:
        raise ValueError("Expected the complete 378-task MBPP+ v0.2.0 release")
    output = Path(__file__).resolve().parent / "data" / "eval.jsonl"
    # Preserve raw serialized inputs; official deserialization happens during scoring.
    write_text(output, text)
    print(f"Wrote {len(rows)} tasks to {output} (MBPP+ {VERSION})")


if __name__ == "__main__":
    main()
