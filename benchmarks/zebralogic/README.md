# ZebraLogic Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official evaluation repository: [WildEval/ZeroEval](https://github.com/WildEval/ZeroEval).
Checked [README](https://github.com/WildEval/ZeroEval/blob/8c1485edf12c6efb5f69135a562927c5ad484059/README.md)
and [src/unified_infer.py](https://github.com/WildEval/ZeroEval/blob/8c1485edf12c6efb5f69135a562927c5ad484059/src/unified_infer.py).

AetherEval retains the documented README profile:
`n=1, temperature=0, top_p=1, max_new_tokens=4096`.
The current generic CLI defaults to 7500 output tokens, illustrating why a
repository URL alone does not specify a unique protocol. This task names the
4096-token README profile rather than silently adopting the generic CLI value.
Model-specific reasoning runs may need a different declared budget; preserve
the grid-mode prompt, answer visibility and full-puzzle scoring when comparing.

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

- Task format follows ZebraLogic (`grid_mode`): each row contains
  `id`, `size`, `puzzle`, and full `solution` table (`header` + `rows`).
- `prepare_data.py` load order:
  1. `allenai/ZebraLogicBench-private/grid_mode` (preferred, gated)
  2. `WildEval/ZebraLogic/grid_mode` (public mirror with answers)
  3. `allenai/ZebraLogicBench/grid_mode` (public fallback, rejected if redacted)
- Local offline file: `data/eval.jsonl`

## Prompting

- Implemented in `task.py`
- Prompt is a ZebraLogic grid prompt with:
  - one worked example,
  - the target puzzle,
  - explicit instruction to return JSON in schema:
    `{"reasoning": "...", "solution": {"House 1": {...}, ...}}`.

## Metrics

- Implemented in `metrics.py`
- Parses the last top-level `{...}` with ZeroEval's brace-stack extractor (newlines
  removed before `json.loads`); unparseable output counts as no answer.
- Compares generated `solution` cells against gold table.
- Reports:
  - `puzzle_accuracy` (all cells correct)
  - `cell_accuracy` (correct cells / total cells over all puzzles, as ZeroEval's `Cell Acc`)
  - `parsed` (JSON parse success rate)
  - difficulty subgroup means (`*_sub_easy`, `*_sub_hard`)
