# AIME25 Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official problem authority: [MAA American Mathematics Competitions](https://maa.org/student-programs/amc/).
The [yentinglin/aime_2025 release](https://huggingface.co/datasets/yentinglin/aime_2025)
is the dataset mirror used here, not an official MAA LLM evaluator.

No official MAA model-generation repository or temperature/top-p/token budget was
identified. The defaults `n=16, temperature=1, top_p=0.7,
max_new_tokens=32768` are an **AetherEval long-reasoning profile**, not an
official AIME protocol. Average accuracy estimates single-sample success;
`pass@k` is not majority-vote accuracy. Report the sampled profile and keep it
fixed across models rather than attributing these settings to the competition.

## Files

```text
benchmarks/aime25/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source dataset: `yentinglin/aime_2025` (split: `train`)
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
