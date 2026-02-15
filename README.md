# AetherEval

A lightweight, generative-only LLM eval framework (For my Own Use :) ).

## Design

- Benchmark root is fixed to `./benchmarks`.
- Task discovery is automatic: any `benchmarks/<task>/task.py + metrics.py` is picked up.
- Backend is offline vLLM only (`dp_size=1` single process, `dp_size>1` Ray data parallel).
- Generation uses vLLM's built-in tqdm progress bar.
- Scoring (`score_generation`) has a framework tqdm progress bar.
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

```bash
aethereval --list-task-defaults
```

Task generation defaults are centrally defined in `configs/task_defaults.yaml`.
You can edit this file to adjust per-task `n/max_new_tokens/temperature/top_p`.
CLI and run YAML still override these defaults.

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
`<model_suffix_lower>`, for example:
`qwen3-0.6b-base`.

If you rerun with the same `run_id`, AetherEval resumes by default from existing `predictions.jsonl`.
Use `--overwrite` to discard old predictions and rerun from scratch.

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
- `load_samples(task_dir) -> list[Sample]`
- `build_prompt(sample) -> str | list[dict]`

`DEFAULT_GEN` is optional in `task.py`; per-task generation defaults are loaded from `configs/task_defaults.yaml`.

Prompt handling:

- Framework defaults to chat-format generation.
- If `build_prompt` returns `str`, it is auto-wrapped to `[{"role":"user","content": ...}]`.
- vLLM uses tokenizer `apply_chat_template`; if unavailable, it falls back to plain `role: content` text and prints a warning.

`metrics.py` must define:

- `score_generation(sample, generation) -> dict` (`score` required)
- `aggregate(sample_results, metric_options) -> dict[str, float]`

Recommended:

- `PRIMARY_METRIC: str` (used by runner to surface report metric in `summary.json`)

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

`summary.json` is task-level aggregate, and includes:

- `metrics`: full metric dict from task aggregate
- `primary_metric`: report metric name
- `primary_score`: report metric value

`run_summary.json` is run-level summary:

- `results`: all per-task summaries
- `primary_scores`: each task's primary metric name/value
- `summary.metrics`: average of same metric names across tasks

## Package Structure

```text
aethereval/
  cli.py
  config.py
  core/
    io.py
    types.py
    task_defaults.py
    task_register.py
    vllm_backend.py
    runner.py
  metrics/
    common.py
    bootstrap.py
configs/
  example.yaml
  task_defaults.yaml
```

## Git LFS

Benchmark JSON data is tracked via `.gitattributes`:

```text
benchmarks/**/data/*.jsonl filter=lfs diff=lfs merge=lfs -text
```

Initialize once in your repo:

```bash
git lfs install
```
