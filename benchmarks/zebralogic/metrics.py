
import json
from collections import defaultdict
from typing import Any

from aethereval.core.types import Sample
from aethereval.metrics.common import mean, to_records


PRIMARY_METRIC = "puzzle_accuracy"


def extract_last_complete_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    best_obj: dict[str, Any] | None = None
    best_end = -1
    best_start = -1

    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        end = start + int(consumed)
        # "last complete JSON object": prefer the one that ends latest.
        # If multiple objects share the same end, keep the outermost one.
        if end > best_end or (end == best_end and (best_start < 0 or start < best_start)):
            best_obj = obj
            best_end = end
            best_start = start

    return best_obj


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

    pred_solution = parsed_obj.get("solution", {}) if isinstance(parsed_obj, dict) else {}
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
    cell_per_sample: list[float] = []
    parsed_per_sample: list[float] = []

    puzzle_by_diff: dict[str, list[float]] = defaultdict(list)
    cell_by_diff: dict[str, list[float]] = defaultdict(list)
    parsed_by_diff: dict[str, list[float]] = defaultdict(list)

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        # Match OLMES behavior: use the first generation record for each sample.
        record = records[0]
        parsed = record.parsed if isinstance(record.parsed, dict) else {}
        sample_puzzle = float(record.score)
        sample_cell = float(parsed.get("cell_accuracy", 0.0))
        sample_parsed = float(parsed.get("parsed", 0.0))
        puzzle_per_sample.append(sample_puzzle)
        cell_per_sample.append(sample_cell)
        parsed_per_sample.append(sample_parsed)

        meta = item.get("meta", {})
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
        "cell_accuracy": mean(cell_per_sample),
        "parsed": mean(parsed_per_sample),
    }

    for difficulty in sorted(puzzle_by_diff.keys()):
        result[f"puzzle_accuracy_sub_{difficulty}"] = mean(puzzle_by_diff[difficulty])
        result[f"cell_accuracy_sub_{difficulty}"] = mean(cell_by_diff[difficulty])
        result[f"parsed_sub_{difficulty}"] = mean(parsed_by_diff[difficulty])

    return result
