# AGIEval English Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [ruixiangcui/AGIEval](https://github.com/ruixiangcui/AGIEval).
Checked [openai_api.py](https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/openai_api.py) and
[run_prediction.py](https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/run_prediction.py).

The reference chat path uses temperature 0 without an explicit output limit;
the completion path uses 2000 tokens. AetherEval uses `n=1, temperature=0,
top_p=1, max_new_tokens=4096`: greedy decoding matches, but 4096 is a local
ceiling, not a universal official limit. The local English MCQ collection and
zero-shot reasoning prompt are an adaptation; do not label it the full official
AGIEval protocol. Official model-specific stop strings are not portable across
chat templates, so this task retains model EOS stopping.

## Files

```text
benchmarks/agieval-en/
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
