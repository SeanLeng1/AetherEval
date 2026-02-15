# MMLU-Pro Benchmark

## Files

```text
benchmarks/mmlu_pro/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source dataset: `TIGER-Lab/MMLU-Pro` (split: `test`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`
- Contains all categories/subsets from MMLU-Pro test split

## Prompting

- Implemented in `task.py`
- Uses dynamic multi-choice format with variable option count (`A` to up to `J`)
- Requires final line format: `Answer: <LETTER>`

## Metrics

- Implemented in `metrics.py`
- Reported metrics include:
  - `accuracy`, `accuracy_stderr`
  - `accuracy@n` when `n>1`
  - `pass@k` with default `k=1,2,4,...,n`
  - `parsed_rate`
  - `accuracy_<category>`
