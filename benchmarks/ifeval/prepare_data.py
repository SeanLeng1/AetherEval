import json
from pathlib import Path

from benchmark_utils.data import read_text, write_text


INPUT_DATA_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "26d8ccdab6fec61b5c83ad6327ea8bda9e580288/instruction_following_eval/data/input_data.jsonl"
)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"

    kept_lines: list[str] = []
    for line in read_text(INPUT_DATA_URL).splitlines():
        line = line.strip()
        if not line:
            continue
        # Validate each line is proper JSON.
        json.loads(line)
        kept_lines.append(line)

    # Keep upstream bytes: a JSON round trip would rewrite escapes such as \u2019.
    write_text(out_path, "".join(line + "\n" for line in kept_lines))

    print(f"wrote {out_path} rows={len(kept_lines)}")


if __name__ == "__main__":
    main()
