# AGIEval English Benchmark

## Files

```text
benchmarks/agieval_en/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source datasets:
  - `dmayhem93/agieval-aqua-rat`
  - `dmayhem93/agieval-gaokao-english`
  - `dmayhem93/agieval-logiqa-en`
  - `dmayhem93/agieval-lsat-ar`
  - `dmayhem93/agieval-lsat-lr`
  - `dmayhem93/agieval-lsat-rc`
  - `dmayhem93/agieval-sat-en`
  - `dmayhem93/agieval-sat-en-without-passage`
  - `dmayhem93/agieval-sat-math`
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

## Prompting

- Implemented in `task.py`
- Converts AGIEval query to clean MCQ prompt with explicit options
- Uses the target ending:
  - `Therefore, the answer is (LETTER)`
  - plus concise reasoning before the final line

## Metrics

- Implemented in `metrics.py`
- Generation-text extraction only (regex parsing of model output); no choice loglikelihood scoring.
- Extraction uses regex priority:
  - exact `Therefore, the answer is (X)`
  - template fallbacks (`so the answer is ...`, `answer: ...`, etc.)
  - final raw letter fallback
- Reported metrics include:
  - `accuracy`, `accuracy_stderr`
  - `accuracy@n` when `n>1`
  - `pass@k` with default `k=1,2,4,...,n`
  - `parsed_rate`
  - `accuracy_<subset>`
