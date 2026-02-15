# GPQA Diamond Benchmark

## Files

```text
benchmarks/gpqa_diamond/
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
