# AetherEval

A lightweight, generative-only LLM evaluation framework.

## Design

- Benchmark root is fixed to `./benchmarks`.
- Task discovery is automatic: any `benchmarks/<task>/task.py + metrics.py` is picked up.
- Backends are offline vLLM and SGLang.
- vLLM supports `dp_size=1` single process and `dp_size>1` Ray data parallel.
- SGLang supports `dp_size=1` single process and `dp_size>1` Ray data parallel.
- Scoring (`score_generation`) has a framework tqdm progress bar.
- Metrics may opt into batch scoring with `score_generations_batch`, used for
  model-based metrics such as reward-model evaluation.
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
  --backend vllm \
  --model Qwen/Qwen3-0.6B-Base \
  --tasks <task_name> \
  --output-dir outputs \
  --max-new-tokens 256
```

`dp-size` and `tp-size` default to `1`, so you only need to set them when overriding.
If `--run-id` is not provided, the default is:
`<model_suffix_lower>`, for example:
`qwen3-0.6b-base`.

Outputs are grouped by model. Without `--run-id`, results are written to
`<output-dir>/<model_suffix>/`; an explicit run id is written to
`<output-dir>/<model_suffix>/<run-id>/`.

If you rerun with the same `run_id`, AetherEval resumes by default from existing `predictions.jsonl`.
Use `--overwrite` to discard old predictions and rerun from scratch.

To use SGLang:

```bash
aethereval \
  --backend sglang \
  --model Qwen/Qwen3-0.6B-Base \
  --tasks <task_name> \
  --output-dir outputs \
  --dp-size 1 \
  --tp-size 1 \
  --context-length 16384 \
  --sglang-generation-batch-size 128 \
  --mem-fraction-static 0.8
```

Backend-specific kwargs can be passed with repeatable key/value flags:

```bash
aethereval --backend vllm --vllm-arg trust_remote_code=true ...
aethereval --backend sglang --sglang-arg trust_remote_code=true ...
```

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
- Offline backends render prompts with tokenizer `apply_chat_template`; if unavailable, the framework falls back to plain `role: content` text and prints a warning.

`metrics.py` must define:

- `score_generation(sample, generation) -> dict` (`score` required)
- `aggregate(sample_results, metric_options) -> dict[str, float]`

Recommended:

- `PRIMARY_METRIC: str` (used by runner to surface report metric in `summary.json`)
- `score_generations_batch(samples, generation_outputs, metric_options) -> list[list[dict]]`
  for metrics that must score generations in batches. The returned outer list must
  align with `generation_outputs`; each inner list must align with that output's
  `generations`.

Shared benchmark implementation code lives in `benchmark_utils/`, outside
`benchmarks/`, so helper modules are not visually or programmatically mixed with
task folders.

## Reward-Model Metrics

RM-based native tasks can receive reward model paths through shared metric flags:

```bash
aethereval \
  --model /path/to/policy \
  --tasks safe_alignment \
  --output-dir outputs
```

`safe_alignment` defaults to `Rihong/Qwen2.5-7B-SafeRLHF-RM` and
`Rihong/Qwen2.5-7B-SafeRLHF-CM`; pass `--rm-model-path` and `--cm-model-path`
only when overriding with local checkpoints.

Optional RM metric flags include `--rm-batch-size`, `--rm-max-length`,
`--rm-device`, `--rm-dtype`, and `--rm-trust-remote-code`.

## External Benchmarks

Some benchmarks do not fit the native `task.py`/`metrics.py` contract because they own
their own generation loop, agent runtime, or reference output layout. These live under
`benchmarks/<name>/` with an `external.py` API. The CLI task router still lets you
select them with `--tasks`; it dispatches them to their external runner internally.

API-Bank is a native task and should be run with `--tasks apibank`.

Current external benchmarks:

- `benchmarks/bfcl` — BFCL-v3 wrapper:
  `aethereval --tasks bfcl --model <model> --output-dir outputs`

External runs use the regular `aethereval` CLI for shared runtime flags
(`--backend`, `--tp-size`, `--gpu-memory-utilization`, `--max-model-len`, etc.) plus
benchmark-specific selectors such as `--categories` for BFCL.

For BFCL with SGLang, `--dp-size > 1` uses SGLang Model Gateway with cache-aware
routing; `--tp-size` remains the tensor-parallel size per replica. This avoids the
upstream BFCL behavior that treats the total GPU count as tensor parallelism.

External benchmark modules use the same shape:

- `ExternalRunSpec`
- `ExternalResult`
- `run(spec) -> ExternalResult`

They still write an AetherEval-style `summary.json` with `metrics`,
`primary_metric`, and `primary_score`, but raw outputs follow each reference
benchmark schema:

```text
outputs/<run_id>/bfcl/
  predictions.jsonl
  result/
  score/
  summary.json
```

See `benchmarks/apibank/README.md` and `benchmarks/bfcl/README.md` for exact metrics,
runtime requirements, and output details.

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
- `meta` (always includes `prompt_token_count` and `response_token_count`)

`summary.json` is task-level aggregate, and includes:

- `metrics`: full metric dict from task aggregate
- `metrics.avg_prompt_tokens` / `metrics.avg_response_tokens`: model-tokenized
  average prompt and response lengths
- `token_usage`: average and total prompt/response token counts
- `primary_metric`: report metric name
- `primary_score`: report metric value

`run_summary.json` is run-level summary:

- `results`: all per-task summaries
- `primary_scores`: each task's primary metric name/value
- `primary_score_aggregate`: mean of task `primary_score` values (direct average across tasks)
- `summary.metrics`: average of same metric names across tasks

## Package Structure

```text
aethereval/
  cli.py
  config.py
  backends/
    base.py
    factory.py
    prompt.py
    sglang/
      backend.py
    vllm/
      backend.py
  core/
    io.py
    types.py
    task_defaults.py
    task_register.py
    runner.py
  metrics/
    common.py
    bootstrap.py
benchmarks/
  <task>/
    README.md
    data/*.jsonl
    task.py
    metrics.py
benchmark_utils/
  aime.py
  instruction_following.py
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
