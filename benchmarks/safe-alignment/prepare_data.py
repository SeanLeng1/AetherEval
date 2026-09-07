import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path("/tmp/GD2PO/safe-alignment/dataset")
DATA_FILE = "data/eval.jsonl"
SOURCE_URL = (
    "https://github.com/Qwen-Applications/GD2PO/tree/main/safe-alignment/dataset"
)


def main() -> None:
    source_root = Path(
        os.environ.get("GD2PO_SAFE_ALIGNMENT_DATA_ROOT", str(DEFAULT_SOURCE_ROOT))
    )
    task_dir = Path(__file__).resolve().parent
    prepare_safe_alignment_data(source_root, task_dir)


def prepare_safe_alignment_data(source_root: Path, task_dir: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for safe_alignment prepare_data.py."
        ) from exc

    datasets = [
        ("alpaca_prompt_only", "alpaca_prompt_only/test.parquet"),
        ("anthropic_hh_rlhf", "anthropic_hh_rlhf/test.parquet"),
        ("pku_saferlhf", "pku-saferlhf/test.parquet"),
    ]

    rows: list[dict[str, Any]] = []
    for prefix, relative_path in datasets:
        parquet_path = source_root / relative_path
        table = pq.read_table(parquet_path)
        for idx, row in enumerate(table.to_pylist()):
            sample_id = f"{prefix}_{idx:05d}"
            rows.append(
                {
                    "id": sample_id,
                    "data_source": row["data_source"],
                    "prompt": _normalize_messages(row["prompt"], sample_id),
                    "ability": row["ability"],
                    "reward_model": row["reward_model"],
                    "extra_info": row["extra_info"],
                    "source": SOURCE_URL,
                    "source_file": str(parquet_path),
                }
            )

    out_path = task_dir / DATA_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


def _normalize_messages(raw: Any, sample_id: str) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompt must be a non-empty list for sample {sample_id}")

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"prompt[{idx}] must be an object for sample {sample_id}")
        role = str(item["role"]).strip()
        content = str(item["content"])
        if not role:
            raise ValueError(f"prompt[{idx}].role is empty for sample {sample_id}")
        messages.append({"role": role, "content": content})
    return messages


if __name__ == "__main__":
    main()
