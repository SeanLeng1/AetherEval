# LiveCodeBench Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).
Checked [runner/parser.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/runner/parser.py)
and [vllm_runner.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/runner/vllm_runner.py).

Defaults follow the reference code-generation runner:
`n=10, temperature=0.2, top_p=0.95, max_new_tokens=2000, stop=["###"]`.
The README also explicitly specifies `n=10, temperature=0.2`.
Reasoning-model evaluations may use longer budgets; those must be reported as
overrides, not conflated with this reference profile.

The local `lighteval/code_generation_lite` v6 snapshot is not automatically the
official runner's `release_latest` set. Prompt formatting, date window and local
execution limits must also match before comparing against a leaderboard score.

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
- Uses a chat prompt with:
  - system: expert Python programmer instruction
  - user sections: `### Question`, `### Format`, `### Answer`
  - includes a concise-reasoning instruction, adapted from OLMES's thinker
    prompting without requiring `<think>` tags; this differs from the official
    LiveCodeBench generic prompt
  - with starter code: complete the provided stub
  - without starter code: read from `stdin`, write to `stdout`
- The framework applies the model chat template.

## Metrics

- Implemented in `metrics.py`
- Runtime executor is benchmark-local in `lcb_eval_runtime.py` (call-based + stdio execution).
- Code extraction uses fenced-code parsing (last fenced block; no raw-text fallback).
- Per generation score:
  - `1.0` if all tests pass
  - `0.0` otherwise

Reported metrics:
- `accuracy`, `accuracy_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
- `parsed_rate`
- `accuracy_<platform>` (e.g., `accuracy_atcoder`, `accuracy_leetcode`)
