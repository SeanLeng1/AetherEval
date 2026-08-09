import json
import re
from pathlib import Path
from typing import Any


SOURCE = "https://github.com/Qwen-Applications/GD2PO/tree/main/tool-calling/API_Bank"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = DATA_DIR / "eval.jsonl"

# ToolRL's rlla data (the training distribution) always pads dialogue-history
# tags with spaces: "<user> ... </user>". The GD2PO source mixes padded and
# unpadded renderings; normalize so eval measures the task, not the tag style.
_HISTORY_TAGS = ("user", "response", "obs")


def _normalize_history_spacing(text: str) -> str:
    for tag in _HISTORY_TAGS:
        text = re.sub(rf"<{tag}>(?=\S)", f"<{tag}> ", text)
        text = re.sub(rf"(?<=\S)</{tag}>", f" </{tag}>", text)
    return text


def _load_level(level: int) -> list[dict[str, Any]]:
    path = DATA_DIR / f"level-{level}-api_processed.json"
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"APIBank source file must contain a list: {path}")
    return rows


def main() -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for level in (1, 2, 3):
            for idx, row in enumerate(_load_level(level)):
                sample = {
                    "id": f"Level{level}_{idx}",
                    "level": level,
                    "source_index": idx,
                    "system": row["system"],
                    "user": _normalize_history_spacing(row["user"]),
                    "answer": row["answer"],
                    "other": row["other"],
                    "source": SOURCE,
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
