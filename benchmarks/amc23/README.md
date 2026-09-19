# AMC23 Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official problem authority: [MAA American Mathematics Competitions](https://maa.org/student-programs/amc/).
The local [RLLab/eval-set](https://huggingface.co/datasets/RLLab/eval-set) subset
is a repackaged problem collection, not an official MAA evaluation repository.

No official MAA LLM decoding protocol was identified. Defaults
`n=16, temperature=0.6, top_p=0.95, max_new_tokens=32768` are our
long-reasoning profile. Numeric-answer extraction with math-verify differs
from administering the original multiple-choice contest. Report this local
variant and distinguish mean sampled accuracy from `pass@k`.

## Data

- Source dataset: `RLLab/eval-set` (config: `amc23`, split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Rows keep the source `problem` and `solution` fields. The `solution` field is numeric in
this subset and is used directly as the gold answer for `math-verify`.

## Metrics

- Primary metric: `accuracy`
- Scored with shared `math-verify` logic from `benchmark_utils/`
- Reports `accuracy@n`, `pass@k`, and parsed-rate metrics when multiple generations are used
