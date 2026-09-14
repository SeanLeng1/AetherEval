# OlympiadBench Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [OpenBMB/OlympiadBench](https://github.com/OpenBMB/OlympiadBench).
Checked the [text-only GPT-4 evaluator](https://github.com/OpenBMB/OlympiadBench/blob/ba5b26a7e2849940b598a9159c1190daa2b9175f/inference/code/evaluators/text_only_gpt_4.py)
and the released model-specific evaluators.

The GPT-4 path sets `temperature=0, max_tokens=2048`; other model paths are
different, so this is not a universal cross-model sampling mandate.
Defaults select that reference: `n=1, temperature=0, top_p=1,
max_new_tokens=2048` (top-p is made explicit; the API reference omits it).
Long-reasoning models may need an explicitly reported budget override.
The local text subset and math-verify scoring do not
reproduce the full bilingual, multimodal OlympiadBench suite. Report the subset
and generation budget explicitly.

## Data

- Source dataset: `RLLab/eval-set` (config: `olympiadbench`, split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Rows keep the source `problem` and `solution` fields. The full `solution` text is used
directly as the gold answer for `math-verify`.

## Metrics

- Primary metric: `accuracy`
- Scored with shared `math-verify` logic from `benchmark_utils/`
- Reports `accuracy@n`, `pass@k`, and parsed-rate metrics when multiple generations are used
