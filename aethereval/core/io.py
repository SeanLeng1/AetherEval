import json
import re
from pathlib import Path
from typing import Any, Iterable


def model_output_name(model: str, model_name: str | None = None) -> str:
    """Return the safe output-directory name for a model.

    An explicit model name is treated as a logical label, not as a path, so slashes
    are sanitized instead of dropping everything before the final component.
    """
    raw_name = (
        str(model).rstrip("/").split("/")[-1]
        if model_name is None
        else str(model_name)
    )
    safe_name = re.sub(r"[^a-z0-9._-]+", "-", raw_name.strip().lower()).strip("-")
    return safe_name or "model"


def default_run_id_for_model(model: str) -> str:
    return model_output_name(model)


def run_output_dir(
    output_dir: str | Path,
    model: str,
    run_id: str | None,
    model_name: str | None = None,
) -> Path:
    """Return ``output/model_name[/run_id]`` without duplicating the default id."""
    model_dir = Path(output_dir) / model_output_name(model, model_name)
    return model_dir / str(run_id) if run_id else model_dir


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

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
