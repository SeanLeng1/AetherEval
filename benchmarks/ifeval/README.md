# IFEval Benchmark

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
