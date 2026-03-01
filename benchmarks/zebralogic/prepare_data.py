
import json
from pathlib import Path
from typing import Any


DATASET_CANDIDATES = [
    ("allenai/ZebraLogicBench-private", "grid_mode"),  # preferred, if gated access is granted
    ("WildEval/ZebraLogic", "grid_mode"),  # public mirror with solutions
    ("allenai/ZebraLogicBench", "grid_mode"),  # public but often redacted
]


def _has_non_redacted_solutions(ds: Any) -> bool:
    for row in ds:
        solution = row.get("solution", {})
        if not isinstance(solution, dict):
            continue
        rows = solution.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row_values in rows:
            if not isinstance(row_values, list):
                continue
            for value in row_values[1:]:
                text = str(value).strip()
                if text and text != "___":
                    return True
    return False


def _load_best_dataset(load_dataset: Any) -> tuple[Any, str, str]:
    failures: list[str] = []
    for dataset_path, dataset_name in DATASET_CANDIDATES:
        try:
            ds = load_dataset(dataset_path, dataset_name, split="test")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{dataset_path}/{dataset_name}: {type(exc).__name__}: {exc}")
            continue
        if not _has_non_redacted_solutions(ds):
            failures.append(f"{dataset_path}/{dataset_name}: solution is redacted (all ___)")
            continue
        return ds, dataset_path, dataset_name

    joined = "\n".join(failures)
    raise RuntimeError(
        "Failed to find a usable ZebraLogic dataset with gold solutions.\n"
        "Tried:\n"
        f"{joined}\n\n"
        "If gated access is required, set HF_TOKEN and retry."
    )


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

    ds, dataset_path, dataset_name = _load_best_dataset(load_dataset)
    print(f"using dataset: {dataset_path}/{dataset_name} rows={len(ds)}")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        sample_id = str(row.get("id", f"zebralogic_{idx:05d}")).strip()
        size = str(row.get("size", "")).strip()
        puzzle = str(row.get("puzzle", "")).strip()
        solution = row.get("solution")
        if not sample_id:
            raise ValueError(f"Missing id at row {idx}")
        if not size:
            raise ValueError(f"Missing size at row {idx}")
        if not puzzle:
            raise ValueError(f"Missing puzzle at row {idx}")
        if not isinstance(solution, dict):
            raise ValueError(f"Missing/invalid solution at row {idx}")

        rows.append(
            {
                "id": sample_id,
                "size": size,
                "puzzle": puzzle,
                "solution": solution,
                "created_at": str(row.get("created_at", "")).strip(),
                "source_dataset": dataset_path,
                "subset": dataset_name,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
