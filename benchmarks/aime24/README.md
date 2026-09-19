# AIME24 Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official problem authority: [MAA American Mathematics Competitions](https://maa.org/student-programs/amc/).
The dataset mirror used here is [RLLab/eval-set](https://huggingface.co/datasets/RLLab/eval-set),
config `aime24`, with transcriptions from MathArena's
[AIME I](https://huggingface.co/datasets/MathArena/aime_2024_I) and
[AIME II](https://huggingface.co/datasets/MathArena/aime_2024_II).
This is not an official MAA LLM evaluator.

No official MAA model-generation repository or temperature/top-p/token budget was
identified. The defaults `n=16, temperature=1, top_p=0.7,
max_new_tokens=32768` are an **AetherEval long-reasoning profile**, not an
official AIME protocol. Average accuracy estimates single-sample success;
`pass@k` is a different statistic, and neither is majority-vote accuracy.
Keep the complete sampling tuple and question set fixed when comparing models.

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

- Source dataset: `RLLab/eval-set`, config `aime24` (split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`
- Run root `push.py --push` before rebuilding local data to publish the MathArena transcriptions.
- All 30 exam questions, row order and gold strings are retained; transcription fixes change prompt text.
- Asymptote/TikZ diagram code is removed during construction; prose, formulas and tables are retained. Use fresh generations after rebuilding.

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
