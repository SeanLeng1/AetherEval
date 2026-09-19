# MMLU-Pro Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro).
Checked [evaluate_from_local.py](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_local.py)
and [evaluate_from_api.py](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_api.py).

Defaults follow the original local runner:
`n=1, temperature=0, top_p=1, max_new_tokens=2048`. The runner's
`stop=["Question:"]` only ends few-shot continuation and is not used with this
zero-shot chat prompt. Unparseable answers score 0 (upstream guesses randomly).
Upstream API branches use different limits (including 4000), so 2048 names a
specific reference profile rather than every official model configuration.

**Prompt mismatch remains:** the official local runner defaults to five
validation-set CoT exemplars; this task currently uses zero-shot CoT. Sampling
alignment does not make these results official five-shot MMLU-Pro scores.

## Files

```text
benchmarks/mmlu-pro/
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
