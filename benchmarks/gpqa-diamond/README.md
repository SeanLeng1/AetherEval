# GPQA Diamond Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [idavidrein/gpqa](https://github.com/idavidrein/gpqa).
Our data/prompt reference is [OpenAI simple-evals](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/gpqa_eval.py).
The [original baseline](https://github.com/idavidrein/gpqa/blob/56686c06f5e19865c153de0fdb11be3890014df7/baselines/run_baseline.py)
has distinct answer-only, CoT and self-consistency paths.

There is no single sampling tuple shared by those paths and all simple-evals
model profiles. Current `n=32, temperature=0.6, top_p=0.95,
max_new_tokens=32768` is our long-reasoning profile, not an official default.
Simple-evals' CLI normally repeats GPQA ten times and reshuffles options;
AetherEval samples multiple answers under each prepared, fixed option ordering.
Consequently equal sample counts would not by themselves reproduce its protocol.

## Files

```text
benchmarks/gpqa-diamond/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source CSV: `https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv`
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

`prepare_data.py` converts source rows to AetherEval JSONL and applies deterministic option shuffling (`random.Random(0)`) per sample.

## Prompting

- Implemented in `task.py`
- Uses instruction-following GPQA format:
  - Ask model to end with `Answer: <LETTER>`
  - Four options shown as `A) ... D) ...`

## Metrics

- Implemented in `metrics.py`
- Deterministic extraction only (no second LLM extraction)
- Choice parser is priority-based (lighteval-style):
  - `final answer ...`
  - `answer: ...`
  - `answer ...`
  - `option/choice ...`
  - line-start choice marker
- No option-text fallback; only extracted choice letters are scored

Reported metrics:

- `accuracy`
- `accuracy_stderr`
- `accuracy@n` (when `n>1`)
- `parsed_rate`
- `pass@k` with default `k=1,2,4,...,n` (doubling schedule)
- `accuracy_<domain>` (if domain exists)

## Notes

- If `n>1`, aggregation first averages each sample over all generated responses, then averages across samples.
