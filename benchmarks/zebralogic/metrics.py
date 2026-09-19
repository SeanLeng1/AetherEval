import json
from collections import defaultdict
from typing import Any

from aethereval.core.types import Sample
from aethereval.metrics.common import mean, to_records


PRIMARY_METRIC = "puzzle_accuracy"


def extract_last_complete_json(text: str) -> dict[str, Any] | None:
    # Port of ZeroEval src/evaluation/eval_utils.py: brace-stack scan for the last
    # top-level {...}, parsed after removing newlines (models often put literal
    # newlines inside the "reasoning" string, which strict JSON rejects).
    stack: list[int] = []
    last_json_start: int | None = None
    last_json_str: str | None = None
    for index, char in enumerate(text):
        if char == "{":
            stack.append(index)
            if last_json_start is None:
                last_json_start = index
        elif char == "}" and stack:
            stack.pop()
            if not stack:
                last_json_str = text[last_json_start : index + 1]
                last_json_start = None

    if last_json_str:
        try:
            obj = json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _normalize_cell(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return ""
        return _normalize_cell(next(iter(value.values())))
    if isinstance(value, list):
        if not value:
            return ""
        return _normalize_cell(value[0])
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    solution_table = sample.gold
    total_cells = int(sample.data.get("total_cells", 0))
    if not isinstance(solution_table, dict) or total_cells <= 0:
        raise ValueError(f"Invalid gold solution in sample {sample.id}")

    parsed_obj = extract_last_complete_json(generation)
    if parsed_obj is None:
        parsed = {
            "parsed": 0.0,
            "cell_accuracy": 0.0,
            "correct_cells": 0,
            "total_cells": total_cells,
            "extract_method": "no_json",
        }
        return {"score": 0.0, "is_pass": False, "parsed": parsed}

    pred_solution = (
        parsed_obj.get("solution", {}) if isinstance(parsed_obj, dict) else {}
    )
    if not isinstance(pred_solution, dict):
        pred_solution = {}
        extract_method = "json_no_solution"
    else:
        extract_method = "json_solution"

    correct_cells = 0
    for house, expected_by_col in solution_table.items():
        if not isinstance(expected_by_col, dict):
            continue
        pred_by_col = pred_solution.get(house, {})
        if not isinstance(pred_by_col, dict):
            pred_by_col = {}
        for column, expected in expected_by_col.items():
            truth_cell = _normalize_cell(expected)
            predicted_cell = _normalize_cell(pred_by_col.get(column))
            if truth_cell and predicted_cell and truth_cell == predicted_cell:
                correct_cells += 1

    cell_accuracy = float(correct_cells) / float(total_cells)
    puzzle_accuracy = 1.0 if correct_cells == total_cells else 0.0
    parsed = {
        "parsed": 1.0,
        "cell_accuracy": cell_accuracy,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "extract_method": extract_method,
    }
    return {
        "score": puzzle_accuracy,
        "is_pass": bool(puzzle_accuracy >= 1.0),
        "parsed": parsed,
    }


def _micro(cells: list[tuple[int, int]]) -> float:
    total = sum(total_cells for _, total_cells in cells)
    return sum(correct for correct, _ in cells) / total if total else 0.0


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    _ = metric_options

    if not sample_results:
        return {
            "puzzle_accuracy": 0.0,
            "cell_accuracy": 0.0,
            "parsed": 0.0,
        }

    puzzle_per_sample: list[float] = []
    # ZeroEval reports Cell Acc as correct_cells / total_cells over all puzzles.
    cell_per_sample: list[tuple[int, int]] = []
    parsed_per_sample: list[float] = []

    puzzle_by_diff: dict[str, list[float]] = defaultdict(list)
    cell_by_diff: dict[str, list[tuple[int, int]]] = defaultdict(list)
    parsed_by_diff: dict[str, list[float]] = defaultdict(list)

    for item in sample_results:
        records = to_records(item["records"])
        if not records:
            continue

        # Match OLMES behavior: use the first generation record for each sample.
        record = records[0]
        parsed = record.parsed if isinstance(record.parsed, dict) else {}
        sample_puzzle = float(record.score)
        sample_cell = (
            int(parsed.get("correct_cells", 0)),
            int(parsed.get("total_cells", 0)),
        )
        sample_parsed = float(parsed.get("parsed", 0.0))
        puzzle_per_sample.append(sample_puzzle)
        cell_per_sample.append(sample_cell)
        parsed_per_sample.append(sample_parsed)

        meta = item["meta"]
        if not isinstance(meta, dict):
            raise ValueError("sample_results meta must be a dict")
        difficulty = str(meta.get("difficulty", "unknown")).strip() or "unknown"
        puzzle_by_diff[difficulty].append(sample_puzzle)
        cell_by_diff[difficulty].append(sample_cell)
        parsed_by_diff[difficulty].append(sample_parsed)

    if not puzzle_per_sample:
        return {
            "puzzle_accuracy": 0.0,
            "cell_accuracy": 0.0,
            "parsed": 0.0,
        }

    result: dict[str, float] = {
        "puzzle_accuracy": mean(puzzle_per_sample),
        "cell_accuracy": _micro(cell_per_sample),
        "parsed": mean(parsed_per_sample),
    }

    for difficulty in sorted(puzzle_by_diff.keys()):
        result[f"puzzle_accuracy_sub_{difficulty}"] = mean(puzzle_by_diff[difficulty])
        result[f"cell_accuracy_sub_{difficulty}"] = _micro(cell_by_diff[difficulty])
        result[f"parsed_sub_{difficulty}"] = mean(parsed_by_diff[difficulty])

    return result
