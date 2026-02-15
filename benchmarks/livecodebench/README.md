# LiveCodeBench Benchmark

## Files

```text
benchmarks/livecodebench/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  lcb_eval_runtime.py
  prepare_data.py
```

## Data

- Source dataset: `lighteval/code_generation_lite` (split: `test`)
- Source subset: `v6`
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Why `v6`:
- This task tracks the latest version-window benchmark (`v6`) and keeps local data size manageable for offline usage.
- As checked on `2026-02-14`, `v6` has `175` rows.

`prepare_data.py` stores compact raw test fields (`public_test_cases` + encoded `private_test_cases`).
`task.py` decodes tests during sample loading and then evaluates fully offline.

## Prompting

- Implemented in `task.py`
- Uses official-style LiveCodeBench code generation instruction:
  - with starter code: complete the provided stub
  - without starter code: read from `stdin`, write to `stdout`
- Framework chat template is applied by AetherEval core.

## Metrics

- Implemented in `metrics.py`
- Runtime executor is benchmark-local in `lcb_eval_runtime.py` (call-based + stdio execution).
- Per generation score:
  - `1.0` if all tests pass
  - `0.0` otherwise

Reported metrics:
- `accuracy`, `accuracy_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
- `parsed_rate`
- `accuracy_<platform>` (e.g., `accuracy_atcoder`, `accuracy_leetcode`)
