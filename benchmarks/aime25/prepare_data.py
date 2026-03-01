
import json
from pathlib import Path


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("datasets is required for prepare_data.py. Install with `pip install datasets`.") from exc

    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("yentinglin/aime_2025", "default", split="train")

    rows: list[dict[str, object]] = []
    for row in ds:
        rows.append(
            {
                "id": str(row["id"]),
                "problem": str(row["problem"]),
                "answer": str(row["answer"]),
                "solution": str(row.get("solution", "")),
                "url": row.get("url"),
                "year": row.get("year"),
                "source": "yentinglin/aime_2025",
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
