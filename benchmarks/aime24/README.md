# AIME24 Benchmark

## Files

```text
benchmarks/aime24/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source dataset: `HuggingFaceH4/aime_2024` (split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

## Prompting

- Implemented in `task.py`
- Uses AetherRL math template style:
  - `{Question}\n\nPlease think step by step, and put your final answer within \boxed{}.`
- Default generation config sets `n=16` (for pass@k style evaluation)

## Metrics

- Implemented in `metrics.py`
- `accuracy` is scored with `math_verify` (AetherRL `math_verify_reward` style)
- If `n>1`, per-sample accuracy is first averaged across responses, then averaged across samples
- If `n>1`, also reports `accuracy@n`
- Includes task-specific `pass@k` with default `k=1,2,4,...,n` (doubling schedule)
