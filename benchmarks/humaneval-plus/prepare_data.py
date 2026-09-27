import gzip
import json
from pathlib import Path
from urllib.request import urlopen

from benchmark_utils.data import write_task_jsonl


HUMANEVAL_PLUS_VERSION = "v0.1.10"
HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    f"{HUMANEVAL_PLUS_VERSION}/HumanEvalPlus.jsonl.gz"
)


def main() -> None:
    with urlopen(HUMANEVAL_PLUS_URL, timeout=120) as response:  # noqa: S310
        raw = response.read()
    text = gzip.decompress(raw).decode("utf-8")

    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        required = {
            "task_id",
            "prompt",
            "entry_point",
            "canonical_solution",
            "base_input",
            "plus_input",
            "atol",
        }
        missing = sorted(required - set(row.keys()))
        if missing:
            raise ValueError(f"Row missing keys: {', '.join(missing)}")
        rows.append(row)

    write_task_jsonl(
        Path(__file__).resolve().parent,
        rows,
        f" source_version={HUMANEVAL_PLUS_VERSION}",
    )


if __name__ == "__main__":
    main()
