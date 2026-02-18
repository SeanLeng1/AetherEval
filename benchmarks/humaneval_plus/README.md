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
- Aligns with OLMES CodexHumanEval(+): uses native EvalPlus `prompt` and appends `Here is the completed function:\n\n```python\n` as the answer prefix.
- Does not append extra format constraints or `contract` hints in the prompt body.
- Core still applies chat template (framework default).

## Metrics

- Implemented in `metrics.py`
- Directly uses EvalPlus runtime (`evalplus.gen.util.trusted_exec` and `evalplus.eval.untrusted_check`).
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
