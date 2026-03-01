
import gzip
import json
from pathlib import Path
from urllib.request import urlopen


HUMANEVAL_PLUS_VERSION = "v0.1.10"
HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    f"{HUMANEVAL_PLUS_VERSION}/HumanEvalPlus.jsonl.gz"
)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(HUMANEVAL_PLUS_URL, timeout=120) as response:  # noqa: S310
        raw = response.read()
    text = gzip.decompress(raw).decode("utf-8")

    rows: list[str] = []
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
        rows.append(json.dumps(row, ensure_ascii=False))

    with out_path.open("w", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")

    print(
        f"wrote {out_path} rows={len(rows)} "
        f"source_version={HUMANEVAL_PLUS_VERSION}"
    )


if __name__ == "__main__":
    main()
