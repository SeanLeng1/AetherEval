
import json
from pathlib import Path
from typing import Any


SOURCE_REPO = "lighteval/code_generation_lite"
SOURCE_SUBSET = "v6"
SOURCE_SPLIT = "test"

def _to_iso_date(value: Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return str(value)
    return str(value or "")


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

    ds = load_dataset(SOURCE_REPO, SOURCE_SUBSET, split=SOURCE_SPLIT)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        question_id = str(row.get("question_id", "")).strip()
        question_content = str(row.get("question_content", "")).strip()
        if not question_id:
            raise ValueError(f"Missing question_id at source row {idx}")
        if not question_content:
            raise ValueError(f"Missing question_content for question_id={question_id}")

        rows.append(
            {
                "id": question_id,
                "question_id": question_id,
                "question_title": str(row.get("question_title", "")).strip(),
                "question_content": question_content,
                "starter_code": str(row.get("starter_code", "")),
                "platform": str(row.get("platform", "")).strip(),
                "difficulty": str(row.get("difficulty", "")).strip(),
                "contest_id": str(row.get("contest_id", "")).strip(),
                "contest_date": _to_iso_date(row.get("contest_date")),
                "public_test_cases": str(row.get("public_test_cases", "")),
                "private_test_cases": str(row.get("private_test_cases", "")),
                "metadata": str(row.get("metadata", "")),
                "source_repo": SOURCE_REPO,
                "source_subset": SOURCE_SUBSET,
                "source_split": SOURCE_SPLIT,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"wrote {out_path} rows={len(rows)} "
        f"(repo={SOURCE_REPO}, subset={SOURCE_SUBSET}, split={SOURCE_SPLIT})"
    )


if __name__ == "__main__":
    main()
