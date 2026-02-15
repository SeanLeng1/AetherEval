from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def default_run_id_for_model(model: str) -> str:
    raw_suffix = str(model).split("/")[-1].strip().lower()
    safe_suffix = re.sub(r"[^a-z0-9._-]+", "-", raw_suffix).strip("-")
    if not safe_suffix:
        safe_suffix = "model"
    return f"{safe_suffix}_{utc_run_id()}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"{path} looks like a Git LFS pointer file. "
                "Run `git lfs install && git lfs pull` to fetch benchmark data."
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
