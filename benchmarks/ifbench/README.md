# IFBench Benchmark

## Files

```text
benchmarks/ifbench/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
  prepare_nltk.py
  ifbench_lib/
```

## Data

- Source benchmark: `allenai/IFBench` official test split (`IFBench_test.jsonl`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`
- NLTK preload script (optional but recommended for offline): `prepare_nltk.py`
- Subset choice: full official test set (294 prompts)

## Prompting

- Implemented in `task.py`
- Prompt is the raw IFBench prompt text.

## Metrics

- Implemented in `metrics.py`
- Uses vendored official IFBench evaluation code in `ifbench_lib/`
- Uses local `ifbench_lib/.nltk_data` first; if missing, falls back to system NLTK paths.
- Strict/loose scoring is kept consistent with `run_eval.py` + `evaluation_lib.py`
- Reported metrics:
  - `prompt_level_strict_acc`
  - `prompt_level_strict_acc_stderr`
  - `inst_level_strict_acc`
  - `inst_level_strict_acc_stderr`
  - `prompt_level_loose_acc`
  - `prompt_level_loose_acc_stderr`
  - `inst_level_loose_acc`
  - `inst_level_loose_acc_stderr`
