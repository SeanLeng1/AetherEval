from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aethereval.core.io import read_jsonl
from aethereval.core.types import Sample


TASK_NAME = "zebralogic"
DATA_FILE = "data/eval.jsonl"

EASY_SIZES = {"2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3"}

ZEBRA_GRID_PROMPT = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def _build_solution_table(solution: dict[str, Any], sample_id: str) -> tuple[dict[str, dict[str, str]], int]:
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or len(header) < 2:
        raise ValueError(f"Invalid solution.header for sample {sample_id}")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid solution.rows for sample {sample_id}")
    if str(header[0]).strip().lower() != "house":
        raise ValueError(f"solution.header[0] must be 'House' for sample {sample_id}")

    columns = [str(x).strip() for x in header]
    solution_table: dict[str, dict[str, str]] = {}
    total_cells = 0
    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"Invalid solution row type for sample {sample_id}, row={i}")
        if len(row) < len(columns):
            raise ValueError(
                f"Solution row length mismatch for sample {sample_id}, row={i}: "
                f"expected >= {len(columns)}, got {len(row)}"
            )
        house_key = f"House {i + 1}"
        solution_table[house_key] = {}
        for j in range(1, len(columns)):
            value = str(row[j]).strip()
            if not value:
                raise ValueError(f"Empty solution cell for sample {sample_id}, house={house_key}")
            solution_table[house_key][columns[j]] = value
            total_cells += 1

    return solution_table, total_cells


def load_samples(task_dir: Path) -> list[Sample]:
    rows = read_jsonl(task_dir / DATA_FILE)

    samples: list[Sample] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ZebraLogic row must be a JSON object")

        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError("Missing sample id")

        size = str(row.get("size", "")).strip()
        puzzle = str(row.get("puzzle", "")).strip()
        solution = row.get("solution")
        if not size:
            raise ValueError(f"Missing size for sample {sample_id}")
        if not puzzle:
            raise ValueError(f"Empty puzzle for sample {sample_id}")
        if not isinstance(solution, dict):
            raise ValueError(f"Missing/invalid solution for sample {sample_id}")

        solution_table, total_cells = _build_solution_table(solution, sample_id)
        difficulty = "easy" if size in EASY_SIZES else "hard"

        samples.append(
            Sample(
                id=sample_id,
                gold=solution_table,
                meta={
                    "difficulty": difficulty,
                    "size": size,
                    "source_dataset": str(row.get("source_dataset", "")).strip(),
                },
                data={
                    "puzzle": puzzle,
                    "size": size,
                    "solution_header": solution.get("header", []),
                    "solution_table": solution_table,
                    "total_cells": total_cells,
                },
            )
        )

    return samples


def build_prompt(sample: Sample) -> str:
    puzzle = str(sample.data["puzzle"]).strip()
    header_raw = sample.data.get("solution_header", [])
    if not isinstance(header_raw, list) or len(header_raw) < 2:
        raise ValueError(f"Invalid solution_header in sample {sample.id}")

    columns = [str(x).strip() for x in header_raw]
    solution_table = sample.data.get("solution_table", {})
    if not isinstance(solution_table, dict) or not solution_table:
        raise ValueError(f"Invalid solution_table in sample {sample.id}")

    json_template: dict[str, Any] = {"reasoning": "___", "solution": {}}
    for i in range(len(solution_table)):
        json_template["solution"][f"House {i + 1}"] = {col: "___" for col in columns[1:]}

    return (
        ZEBRA_GRID_PROMPT.replace("{puzzle}", puzzle).replace(
            "{json_template}", json.dumps(json_template, indent=4)
        )
    )
