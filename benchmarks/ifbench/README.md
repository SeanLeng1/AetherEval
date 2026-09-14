# IFBench Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [allenai/IFBench](https://github.com/allenai/IFBench).
Checked [README evaluation note](https://github.com/allenai/IFBench/blob/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/README.md)
and [config.py](https://github.com/allenai/IFBench/blob/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d/config.py).

The paper protocol stated in the README uses temperature 0, model-dependent
output budgets, and final-answer extraction for thinking models. The newer
utility config instead defaults to temperature 0.6 and 4096 tokens.
AetherEval deliberately follows the stated paper protocol's greedy decoding:
`n=1, temperature=0, top_p=1, max_new_tokens=4096`.
The 4096 ceiling is a local non-thinking budget, not the paper's universal limit.
For thinking models, also verify the output budget and reasoning removal before
comparing prompt-level loose accuracy.

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
- `PRIMARY_METRIC`: `prompt_level_loose_acc`
- Reported metrics:
  - `prompt_level_strict_acc`
  - `prompt_level_strict_acc_stderr`
  - `inst_level_strict_acc`
  - `inst_level_strict_acc_stderr`
  - `prompt_level_loose_acc`
  - `prompt_level_loose_acc_stderr`
  - `inst_level_loose_acc`
  - `inst_level_loose_acc_stderr`
