from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets is required for prepare_data.py. Install with `pip install datasets`."
        ) from exc

    task_dir = Path(__file__).resolve().parent
    out_path = task_dir / "data" / "eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("jgyasu/bbeh", "zebra_puzzles", split="train")

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(ds):
        question = str(row["input"]).strip()
        answer = str(row["target"]).strip()
        if not question or not answer:
            raise ValueError(f"Invalid zebra_puzzles row at index {idx}")

        rows.append(
            {
                "id": f"zebra_{idx:05d}",
                "question": question,
                "answer": answer,
                "subset": "zebra_puzzles",
                "source": "jgyasu/bbeh",
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
