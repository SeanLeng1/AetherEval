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
- Subset choice: full official test set (300 prompts)

### Upstream data correction (verified 2026-09-14)

The previous 294-row local artifact was **byte-identical to the original
upstream release**, commit
[`8a66c06`](https://github.com/allenai/IFBench/blob/8a66c069755e98b8d0871829751da9fd7b6ce8f0/data/IFBench_test.jsonl).
That release embedded questions 269–274 inside question 268's prompt and gave
question 268 the `custom:csv_city` instruction from question 274 instead of
`words:vowel`. This was inherited from upstream, not introduced by AetherEval's
preparation or evaluation code. The malformed record was still valid JSON, so
JSON parsing alone did not detect the problem.

Upstream fixed the file on **2026-04-11** in
[`cb932e3` — fix test file](https://github.com/allenai/IFBench/commit/cb932e352a505306ad0115272211df14bb8f628f),
restoring all 300 questions. This branch's data was reprocessed from the official
JSONL on 2026-09-14. The embedded tabs and row fields suggest an earlier tabular
conversion/quoting error, but the original conversion mechanism has not been
verified.

Results produced with the old artifact describe the defective 294-question
release, not the corrected 300-question benchmark. Updating them requires new
generations for corrected question 268 and missing questions 269–274; rescoring
old generations alone is insufficient. Other generations may be reused after
checking IDs and prompt compatibility. Recompute the aggregate and any dependent
checkpoint selection or tables; unchanged rankings must not be assumed.
The `main` branch and existing result files were not modified by this refresh.

## Prompting

- Implemented in `task.py`
- Prompt is the raw IFBench prompt text.

## Metrics

- Implemented in `metrics.py`
- Uses vendored official IFBench evaluation code in `ifbench_lib/`
- Checker source pinned to upstream
  [`1c40f0c` (2026-09-09)](https://github.com/allenai/IFBench/tree/1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d),
  synchronized on 2026-09-14; syllable counting uses `syllapy==0.8.0`.
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

### Checker synchronization (2026-09-14)

This refresh includes the upstream verifier fixes, not just the 300-row data
correction. In particular,
[`1d62c1d`](https://github.com/allenai/IFBench/commit/1d62c1d6dc4f6608938d22424c46edef6ce50671)
fixes whole-word keyword counts, word-trigram overlap, inclusive word-span
copying, punctuation handling in word positions and alphabetical sentence
starts, and enforcement of word uniqueness. Earlier upstream fixes to
tokenization, bounds/empty-input handling, pronouns and other verifiers are
also included. These changes can alter scores for unchanged generations.

Local adaptations are limited to relative imports/formatting, the existing
offline NLTK resource path, and the native per-sample scoring interface. The
58 IFBench OOD registry entries are retained; upstream's additional classic
IFEval registry and CLI/reporting code are not imported. Strict/loose scoring
and missing-response/whitespace lookup behavior follow the pinned upstream
functions. NLTK resources must be installed with `prepare_nltk.py` before an
offline run; scoring no longer attempts automatic downloads on import.

Verification: all 65 upstream IFBench `InstructionsTest` tests pass against
the synchronized local verifiers (the separate classic IFEval tests are not
part of this benchmark). This does not establish that every verifier is free
of bugs. Rescore saved generations with this version before comparing new and
old runs, and regenerate the seven data-correction cases described above.
Historical result files and the `main` branch are unchanged.
