
import json
from pathlib import Path


SOURCE_DATA = Path("/root/IFBench/data/IFBench_test.jsonl")


def main() -> None:
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(
            f"IFBench source data not found: {SOURCE_DATA}. "
            "Clone IFBench to /root/IFBench first."
        )

    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    with SOURCE_DATA.open("r", encoding="utf-8") as f:
        for line in f:
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
