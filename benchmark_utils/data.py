"""Read preparation inputs and atomically write prepared task data."""

import hashlib
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from aethereval.core.io import write_jsonl


def read_bytes(source: str | Path, sha256: str | None = None) -> bytes:
    if str(source).startswith(("http://", "https://")):
        with urlopen(str(source), timeout=120) as response:
            payload = response.read()
    else:
        payload = Path(source).read_bytes()
    if sha256 is not None and hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"{source} changed upstream (sha256 mismatch); re-audit it")
    return payload


def read_text(source: str | Path, sha256: str | None = None) -> str:
    return read_bytes(source, sha256).decode("utf-8")


def load_hf(repo: str, config: str | None, split: str, revision: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for prepare_data.py. Install with `pip install datasets`."
        ) from exc
    return load_dataset(repo, config, split=split, revision=revision)


def write_task_jsonl(task_dir: Path, rows: list[dict[str, Any]], note: str = "") -> None:
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    print(f"wrote {out_path} rows={len(rows)}{note}")


def write_text(path: Path, text: str) -> None:
    """Atomically replace a file with text kept verbatim (no JSON round trip)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
