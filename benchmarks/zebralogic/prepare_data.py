from pathlib import Path
from typing import Any

from benchmark_utils.data import load_hf, write_task_jsonl


DATASET_CANDIDATES = [
    (
        "allenai/ZebraLogicBench-private",
        "grid_mode",
        "9f39ef490ae924437376657205025f26c0bd1af3",
    ),  # preferred, if gated access is granted
    (
        "WildEval/ZebraLogic",
        "grid_mode",
        "0a473f5a0054835754ed156d5a79c6ce27178bb1",
    ),  # public mirror with solutions
    (
        "allenai/ZebraLogicBench",
        "grid_mode",
        "2f94a445d7079f20146f5443e2606049de8543e0",
    ),  # public but often redacted
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


def _load_best_dataset() -> tuple[Any, str, str]:
    failures: list[str] = []
    for dataset_path, dataset_name, revision in DATASET_CANDIDATES:
        try:
            ds = load_hf(dataset_path, dataset_name, "test", revision)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"{dataset_path}/{dataset_name}: {type(exc).__name__}: {exc}"
            )
            continue
        if not _has_non_redacted_solutions(ds):
            failures.append(
                f"{dataset_path}/{dataset_name}: solution is redacted (all ___)"
            )
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
    ds, dataset_path, dataset_name = _load_best_dataset()
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

    write_task_jsonl(Path(__file__).resolve().parent, rows)


if __name__ == "__main__":
    main()
