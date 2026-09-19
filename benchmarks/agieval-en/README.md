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

- Source: official [ruixiangcui/AGIEval](https://github.com/ruixiangcui/AGIEval) `data/v1_1/<subset>.jsonl`,
  pinned to commit `84ab72d`, for `aqua-rat`, `gaokao-english`, `logiqa-en`, `lsat-ar`,
  `lsat-lr`, `lsat-rc`, `sat-en`, `sat-math` (2646 questions). These are the 8 tasks of
  OLMES `agi_eval_english`; `sat-en-without-passage` is not part of that suite.
  The `dmayhem93/agieval-*` HF mirrors are not used: they lost option (D) of three
  SAT-English questions, one of which is the gold answer.
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
- Primary metric: `macro_accuracy`, the unweighted mean of the 8 subset accuracies
  (OLMES `agi_eval_english` convention).
- Reported metrics include:
  - `accuracy` (micro over questions), `accuracy_stderr`
  - `accuracy@n` when `n>1`
  - `pass@k` with default `k=1,2,4,...,n`
  - `parsed_rate`
  - `accuracy_<subset>`
