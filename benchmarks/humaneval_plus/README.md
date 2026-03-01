# HumanEval+ Benchmark

## Files

```text
benchmarks/humaneval_plus/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source: EvalPlus HumanEval+ release (`v0.1.10`)
- Local offline file: `data/eval.jsonl`
- Regeneration: `prepare_data.py`

Each row keeps EvalPlus fields (`task_id`, `prompt`, `entry_point`, `canonical_solution`, `base_input`, `plus_input`, `atol`, ...).

## Prompting

- Implemented in `task.py`
- Uses chat-style prompt with explicit sections:
  - system instruction for Python code completion
  - user sections: `### Question`, `### Format`, `### Answer`
- `### Format` asks for short reasoning plus completed function inside a fenced Python block.
- The framework applies the model chat template.

## Metrics

- Implemented in `metrics.py`
- For each generation:
  - extract the final code candidate (prefer answer block / fenced code)
  - execute `prompt + continuation`
  - run `test + check(entry_point)` inside a local sandboxed subprocess with timeout
- `score` / `is_pass` is pass/fail of that unit-test execution.

Reported metrics include:
- `accuracy` (alias of `accuracy_plus`)
- `accuracy_plus`, `accuracy_base`
- `accuracy_stderr`, `accuracy_plus_stderr`, `accuracy_base_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
