# BBH Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard).
The [paper, evaluation protocol](https://arxiv.org/html/2210.09261v1#S3) specifies
greedy decoding and three CoT exemplars; the
[released prompts](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/9ee07bd481feebf959a6b59d61ea57bdcf30964d/cot-prompts) are the reference.

AetherEval defaults to `n=1, temperature=0, top_p=1, max_new_tokens=4096`.
The token cap is a local budget, not a limit established by that protocol.
**This task is zero-shot CoT, not the paper's three-shot CoT.** Matching
temperature alone does not make its scores directly comparable with that table.

## Files

```text
benchmarks/bbh/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source dataset: `lukaemon/bbh` (all BBH subsets, split: `test`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`
- First-time setup: run `python benchmarks/bbh/prepare_data.py`

Each row includes: `id`, `subset`, `input`, `target`, `answer`, `description`.

## Prompting

- Implemented in `task.py`
- Uses a zero-shot CoT query format:
  - `Question: <input>`
  - `Answer: Let's think step by step.`
- Prepends subset description when available.
- Uses zero-shot CoT without few-shot exemplars.

## Metrics

- Implemented in `metrics.py`
- Uses generation-text extraction only (no likelihood scoring)
- Uses per-subset answer regex rules
- Extraction never uses the gold answer as a search pattern. MC subsets extract
  choice letters even for the three malformed rows with free-form gold labels.
  These rows remain in the denominator; they are not silently repaired or removed.
- Core metric is exact match with normalization:
  - `ignore_case=True`
  - `ignore_punctuation=True` for all subsets except `dyck_languages`

Reported metrics include:
- `exact_match`, `exact_match_stderr`
- `accuracy`, `accuracy_stderr` (alias-compatible)
- `accuracy@n` / `pass@k` when `n>1`
- `parsed_rate`
- `exact_match_<subset>` and `accuracy_<subset>`
