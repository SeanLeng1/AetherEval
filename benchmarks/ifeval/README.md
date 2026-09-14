# IFEval Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [google-research/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
The [official README](https://github.com/google-research/google-research/blob/master/instruction_following_eval/README.md)
accepts a JSONL of already-generated prompt/response pairs; it does not prescribe
one model-independent temperature/top-p/output limit.

The defaults `n=1, temperature=0, top_p=1, max_new_tokens=4096`
are an explicit local greedy profile. The raw prompts and instruction checkers
come from the release, but the decoding tuple is not an official requirement.
Report whether strict or loose, prompt-level or instruction-level accuracy is
being compared. AetherEval's primary metric is prompt-level loose accuracy.

## Files

```text
benchmarks/ifeval/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  ifeval_lib/
```

## Data

- Upstream source: Google `instruction_following_eval/data/input_data.jsonl`
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`
- NLTK preload script (optional but recommended for offline): `prepare_nltk.py`

## Prompting

- Implemented in `task.py`
- Each sample uses the raw IFEval prompt text directly as generation prompt.

## Metrics

- Implemented in `metrics.py`
- Uses vendored Google evaluation logic in `ifeval_lib/evaluation_lib.py`
- Uses local `ifeval_lib/.nltk_data` first; if missing, falls back to system NLTK paths.
- `PRIMARY_METRIC`: `prompt_level_loose_acc`
- Reports:
  - `prompt_level_strict_acc`
  - `prompt_level_strict_acc_stderr`
  - `inst_level_strict_acc`
  - `inst_level_strict_acc_stderr`
  - `prompt_level_loose_acc`
  - `prompt_level_loose_acc_stderr`
  - `inst_level_loose_acc`
  - `inst_level_loose_acc_stderr`

## Notes

- If `n>1`, aggregation first averages each sample over all generated responses, then averages across samples.
