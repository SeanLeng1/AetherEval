# Minerva Math Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official source: [Minerva paper](https://arxiv.org/html/2206.14858v2), particularly
Sections 2.3–2.5 and the OCWCourses appendix.
No standalone author-maintained OCWCourses inference repository was verified.
The local [RLLab/eval-set/minervamath](https://huggingface.co/datasets/RLLab/eval-set)
contains OCW-style problems; it is not the lm-evaluation-harness
`minerva_math` variant over the MATH dataset.

The paper uses a 512-token generation limit, greedy single-sample evaluation,
and `temperature=0.6, top_p=0.95` for multiple samples (64 for OCW majority
voting). Defaults now select the single-sample path:
`n=1, temperature=0, top_p=1, max_new_tokens=512`.
The local prompt and math-verify scorer still differ from the paper's few-shot
evaluation; this is decoding alignment, not full protocol reproduction.
Long-reasoning evaluations should explicitly override the output budget and report it.

## Data

- Source dataset: `RLLab/eval-set` (config: `minervamath`, split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Rows keep the source `problem` and `solution` fields. The full `solution` text is used
directly as the gold answer for `math-verify`.

## Metrics

- Primary metric: `accuracy`
- Scored with shared `math-verify` logic from `benchmark_utils/`
- Reports `accuracy@n`, `pass@k`, and parsed-rate metrics when multiple generations are used
