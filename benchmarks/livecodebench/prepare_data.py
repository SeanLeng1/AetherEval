from pathlib import Path
from typing import Any

from benchmark_utils.data import load_hf, write_task_jsonl


SOURCE_REPO = "lighteval/code_generation_lite"
SOURCE_SUBSET = "v6"
SOURCE_SPLIT = "test"
SOURCE_REVISION = "89e5fc5c2a8e748f50e95bc7235fab2372d49bfa"


def _to_iso_date(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value or "")


def main() -> None:
    ds = load_hf(SOURCE_REPO, SOURCE_SUBSET, SOURCE_SPLIT, SOURCE_REVISION)

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

    write_task_jsonl(
        Path(__file__).resolve().parent,
        rows,
        f" (repo={SOURCE_REPO}, subset={SOURCE_SUBSET}, split={SOURCE_SPLIT})",
    )


if __name__ == "__main__":
    main()
