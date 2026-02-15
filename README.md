# AetherEval

A lightweight, generative-only LLM eval framework.

## Design

- Benchmark root is fixed to `./benchmarks`.
- Task discovery is automatic: any `benchmarks/<task>/task.py + metrics.py` is picked up.
- Backend is offline vLLM only (`dp_size=1` single process, `dp_size>1` Ray data parallel).
- Single-worker generation uses vLLM's built-in tqdm progress bar.
- Task owns prompt/data/metric logic; core only orchestrates loading, generation, scoring, resume, and output writing.
- Supports `n` sampling; metrics are fully task-defined.

## Install

```bash
source /root/env/bin/activate
pip install -e .
```

## List tasks

```bash
aethereval --list-tasks
```

## Run (single GPU)

```bash
aethereval \
  --model Qwen/Qwen3-0.6B-Base \
  --tasks <task_name> \
  --output-dir outputs \
  --max-new-tokens 256
```

`dp-size` and `tp-size` default to `1`, so you only need to set them when overriding.
If `--run-id` is not provided, the default is:
`<model_suffix_lower>_<YYYYMMDD_HHMMSS>` (UTC), for example:
`qwen3-0.6b-base_20260215_013000`.

## Run With YAML

```bash
aethereval --config configs/example.yaml
```

CLI has higher priority than YAML.

## Inspect Prompts (No Inference)

```bash
aethereval \
  --model Qwen/Qwen3-0.6B-Base \
  --tasks gpqa_diamond \
  --inspect
```

This prints the first 5 prompts after chat-template rendering and exits.

## Benchmark Contract

Each benchmark folder must include:

```text
benchmarks/<task_name>/
  README.md
  data/*.jsonl
  task.py
  metrics.py
```

`task.py` must define:

- `TASK_NAME: str`
- `DATA_FILE: str` (must be `.jsonl`)
- `DEFAULT_GEN: dict`
- `load_samples(task_dir) -> list[Sample]`
- `build_prompt(sample) -> str | list[dict]`

Prompt handling:

- Framework defaults to chat-format generation.
- If `build_prompt` returns `str`, it is auto-wrapped to `[{"role":"user","content": ...}]`.
- vLLM uses tokenizer `apply_chat_template`; if unavailable, it falls back to plain `role: content` text and prints a warning.

`metrics.py` must define:

- `score_generation(sample, generation) -> dict` (`score` required)
- `aggregate(sample_results, metric_options) -> dict[str, float]`

## Bootstrap

Bootstrap options are configured from CLI/YAML and forwarded to each task `aggregate`:

- `--bootstrap-resamples`
- `--bootstrap-seed`
- `--bootstrap-confidence`

Multi-generation behavior:

- If `n=1`, metrics use single-generation scores.
- If `n>1`, metrics aggregate each sample over all generated responses first, then average across samples.
- Task-specific metrics may additionally report `accuracy@n` and `pass@k` (commonly `k=1,2,4,...,n`).

Task-specific details (data source, prompt template, metric definition) should live in each task folder README, e.g. `benchmarks/ifeval/README.md`.

## Output Format

Per run:

```text
outputs/<run_id>/
  run_summary.json
  <task>/
    predictions.jsonl
    summary.json
    run_config.json
```

`predictions.jsonl` contains one row per `(sample_id, gen_idx)`:

- `sample_id`
- `gen_idx`
- `prompt`
- `generation`
- `score`
- `is_pass`
- `parsed`
- `gold`
- `error`
- `meta`

`summary.json` is task-level aggregate.

`run_summary.json` is run-level summary:

- `results`: all per-task summaries
- `summary.metrics`: average of same metric names across tasks

## Git LFS

Benchmark JSON data is tracked via `.gitattributes`:

```text
benchmarks/**/data/*.jsonl filter=lfs diff=lfs merge=lfs -text
```

Initialize once in your repo:

```bash
git lfs install
```
