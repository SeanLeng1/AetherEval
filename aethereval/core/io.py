
import json
import re
from pathlib import Path
from typing import Any, Iterable


def default_run_id_for_model(model: str) -> str:
    raw_suffix = str(model).split("/")[-1].strip().lower()
    safe_suffix = re.sub(r"[^a-z0-9._-]+", "-", raw_suffix).strip("-")
    if not safe_suffix:
        safe_suffix = "model"
    return safe_suffix


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            oid_line = f.readline().strip()
            size_line = f.readline().strip()
            details: list[str] = []
            if oid_line.startswith("oid "):
                details.append(oid_line)
            if size_line.startswith("size "):
                details.append(size_line)
            details_text = f" ({', '.join(details)})" if details else ""
            raise RuntimeError(
                f"{path} is a Git LFS pointer file{details_text}. "
                "Run `git lfs install && git lfs pull` to fetch benchmark data, "
                "then verify with `git lfs ls-files`."
            )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
