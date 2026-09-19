import json
from pathlib import Path

from benchmark_utils.data import read_text


SOURCE_DATA = (
    "https://raw.githubusercontent.com/allenai/IFBench/"
    "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/data/IFBench_test.jsonl"
)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for line in read_text(SOURCE_DATA).split("\n"):
        line = line.strip()
        if not line:
            continue
        json.loads(line)
        lines.append(line)

    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"wrote {out_path} rows={len(lines)}")


if __name__ == "__main__":
    main()
