# MATH500 Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official sources: [hendrycks/math](https://github.com/hendrycks/math) for MATH and
[openai/prm800k](https://github.com/openai/prm800k) for the 500-problem test subset.
A reference evaluator is [OpenAI simple-evals/math_eval.py](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/math_eval.py).

Defaults `n=16, temperature=1, top_p=0.7, max_new_tokens=32768`
are our long-reasoning profile, not a universal MATH500 standard.
Simple-evals selects decoding through model-specific samplers and uses an
LLM equivalence checker; this task instead uses a zero-shot boxed-answer
prompt and math-verify. Matching the problem set does not establish identical
prompting, grading, or sampling. `accuracy@n` here averages samples; it is not
majority voting.

## Data

- Source dataset: `RLLab/eval-set` (config: `math500`, split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Rows keep the source `problem` and `solution` fields. The full `solution` text is used
directly as the gold answer for `math-verify`.

## Metrics

- Primary metric: `accuracy`
- Scored with shared `math-verify` logic from `benchmark_utils/`
- Reports `accuracy@n`, `pass@k`, and parsed-rate metrics when multiple generations are used
