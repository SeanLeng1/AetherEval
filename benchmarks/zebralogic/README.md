# ZebraLogic Benchmark

## Files

```text
benchmarks/zebralogic/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Task format follows OLMES ZebraLogic (`grid_mode`): each row contains
  `id`, `size`, `puzzle`, and full `solution` table (`header` + `rows`).
- `prepare_data.py` load order:
  1. `allenai/ZebraLogicBench-private/grid_mode` (preferred, gated)
  2. `WildEval/ZebraLogic/grid_mode` (public mirror with answers)
  3. `allenai/ZebraLogicBench/grid_mode` (public fallback, rejected if redacted)
- Local offline file: `data/eval.jsonl`

## Prompting

- Implemented in `task.py`
- Prompt is OLMES-style ZebraLogic grid prompt with:
  - one worked example,
  - the target puzzle,
  - explicit instruction to return JSON in schema:
    `{"reasoning": "...", "solution": {"House 1": {...}, ...}}`.

## Metrics

- Implemented in `metrics.py`
- Parses the last complete JSON object in generation output.
- Compares generated `solution` cells against gold table.
- Reports:
  - `puzzle_accuracy` (all cells correct)
  - `cell_accuracy` (fraction of correct cells)
  - `parsed` (JSON parse success rate)
  - difficulty subgroup means (`*_sub_easy`, `*_sub_hard`)
