# HumanEval+ Benchmark

## Files

```text
benchmarks/humaneval_plus/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
  eval_runtime.py
```

## Data

- Source: EvalPlus HumanEval+ release (`v0.1.10`)
- Local offline file: `data/eval.jsonl`
- Regeneration: `prepare_data.py`

Each row keeps EvalPlus fields (`task_id`, `prompt`, `entry_point`, `canonical_solution`, `base_input`, `plus_input`, `atol`, ...).

## Prompting

- Implemented in `task.py`
- Instruction asks model to output executable Python code only.
- Core still applies chat template (framework default).

## Metrics

- Implemented in `metrics.py`
- Uses benchmark-local runtime (`eval_runtime.py`) adapted from EvalPlus execution logic.
- For each generation:
  - run base tests
  - run plus tests only if base passes
- `score` / `is_pass` is based on HumanEval+ criterion (base + plus pass).

Reported metrics include:
- `accuracy` (alias of `accuracy_plus`)
- `accuracy_plus`, `accuracy_base`
- `accuracy_stderr`, `accuracy_plus_stderr`, `accuracy_base_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
