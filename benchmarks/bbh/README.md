# BBH Benchmark

## Files

```text
benchmarks/bbh/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source dataset: `lukaemon/bbh` (all BBH subsets, split: `test`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`
- First-time setup: run `python benchmarks/bbh/prepare_data.py`

Each row includes: `id`, `subset`, `input`, `target`, `answer`, `description`.

## Prompting

- Implemented in `task.py`
- Uses a zero-shot CoT query format:
  - `Question: <input>`
  - `Answer: Let's think step by step.`
- Prepends subset description when available.
- Uses zero-shot CoT without few-shot exemplars.

## Metrics

- Implemented in `metrics.py`
- Uses generation-text extraction only (no likelihood scoring)
- Uses per-subset answer regex rules
- Includes fallback handling for a small number of BBH rows where MC subsets
  provide free-form gold text instead of `(A)/(B)/...`.
- Core metric is exact match with normalization:
  - `ignore_case=True`
  - `ignore_punctuation=True` for all subsets except `dyck_languages`

Reported metrics include:
- `exact_match`, `exact_match_stderr`
- `accuracy`, `accuracy_stderr` (alias-compatible)
- `accuracy@n` / `pass@k` when `n>1`
- `parsed_rate`
- `exact_match_<subset>` and `accuracy_<subset>`
