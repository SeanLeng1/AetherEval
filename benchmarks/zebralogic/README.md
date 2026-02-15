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

- Source: `jgyasu/bbeh` subset `zebra_puzzles` (same subset used in lighteval BBEH)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

## Prompting

- Implemented in `task.py`
- Uses direct puzzle prompt and requests final line format `ANSWER: <answer>`.

## Metrics

- Implemented in `metrics.py`
- Extracts answer from:
  - last `ANSWER:` line, else
  - last non-empty line
- Compares normalized string exact-match.
- Reports:
  - `accuracy`, `accuracy_stderr`
  - `accuracy@n` when `n>1`
  - `pass@k` (`k=1,2,4,...,n` by default)
  - `parsed_rate`
