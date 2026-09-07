# AetherEval

A lightweight, generative-only LLM evaluation framework.

## Design

- Benchmark root is fixed to `./benchmarks`.
- Task discovery is automatic: any `benchmarks/<task>/task.py + metrics.py` is picked up.
- Backends are offline vLLM and SGLang.
- vLLM supports `dp_size=1` single process and `dp_size>1` Ray data parallel.
- SGLang always uses Ray-managed tensor-parallel servers behind the SGLang Model
  Gateway (SMG), including when `dp_size=1`.
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
CLI and run YAML override these defaults. One protocol guard applies: when
`temperature=0` is set globally without an explicit `n`, tasks whose default
`n>1` retain their task-default sampling temperature. Pass `--n 1` as well to
explicitly switch those tasks to single-generation greedy decoding.

## Run (single GPU)

```bash
aethereval \
  --backend vllm \
  --model Qwen/Qwen3-0.6B-Base \
  --tasks <task_name> \
  --output-dir outputs \
  --max-new-tokens 256
```

### Ray data parallelism

Evaluation commands run directly in the invoking shell. SGLang data-parallel
replicas are Ray actors behind one SMG router:

```bash
aethereval --model /path/to/model --tasks apibank --dp-size 8 --tp-size 1
```

For multiple nodes, start and join the Ray cluster manually before invoking
`aethereval` once on the head node. Set `RAY_ADDRESS=auto` when necessary so
the driver connects to that cluster. AetherEval intentionally does not perform
SSH, scheduler allocation, or worker-node joins.

For SGLang DP, Ray places one gRPC server actor per replica, including across
joined worker nodes, and AetherEval starts one SMG router on the driver. SMG
routes requests to the workers over gRPC with its default cache-aware policy.
No SGLang server or router needs to be started manually on any node.
Reward-model (embedding) replicas are the exception: they serve HTTP and are
scored round-robin without SMG, whose gRPC embedding pipeline accepts text only
while reward scoring sends token ids.
Worker gRPC and NCCL-initialization ports are selected independently on the
node where each actor runs. Startup collisions on either port are retried
automatically, and shutdown terminates the complete SGLang process group so
scheduler children cannot retain GPUs or ports.

Each replica is one Ray actor requesting `tp-size` GPUs, so its tensor-parallel
group must fit on one node. For two eight-GPU nodes, `--dp-size 16 --tp-size 1`
and `--dp-size 2 --tp-size 8` are supported topologies; one cross-node
`--tp-size 16` replica is not supported by this launcher.

`dp-size` and `tp-size` default to `1`, so you only need to set them when overriding.
If `--run-id` is not provided, the default is:
`<model_suffix_lower>`, for example:
`qwen3-0.6b-base`.

Use `--model-name` when the model ID/path suffix is not a useful output label. It
changes only the logical/output name; `--model` is still passed unchanged to the
backend for loading:

```bash
aethereval \
  --model qwen2.5/huggingface \
  --model-name qwen2.5_huggingface \
  --tasks <task_name> \
  --output-dir outputs
```

Outputs are grouped by model. Without `--run-id`, results are written to
`<output-dir>/<model-name-or-model-suffix>/`; an explicit run id is written to
`<output-dir>/<model-name-or-model-suffix>/<run-id>/`.

If you rerun with the same `run_id`, AetherEval resumes by default from existing `predictions.jsonl`.
Use `--overwrite` to discard old predictions and rerun from scratch.

A normal native-task run uses the same two phases automatically: it completes
generation for every selected native task, unloads the candidate backend, and
then evaluates every task. This ordering is shared by API judges, local judges,
and non-judge metrics. Explicit phase flags are only needed when the two phases
must run as separate commands or on separate machines.

### Split offline generation from online evaluation

Use `--generate-only` when the inference machine has no network access. This
starts the candidate backend and writes complete, explicitly unscored
`predictions.jsonl` files without initializing metrics or LLM judges:

```bash
aethereval \
  --backend sglang \
  --model /path/to/candidate \
  --model-name candidate \
  --tasks llmeval-med,healthbench,writingbench,creative_writing_v3,researchqa,arena_hard_v2 \
  --output-dir /output \
  --run-id production-1 \
  --dp-size 8 \
  --tp-size 1 \
  --generate-only
```

On a machine with judge API access, mount or copy the output directory and run
`--eval-only`. It validates that every expected `(sample_id, gen_idx)` exists,
then scores and atomically replaces the unscored records. It does not create a
vLLM/SGLang candidate backend, and the `--model` path does not need to exist on
that machine; it is still required to locate the same output directory.

```bash
export AETHEREVAL_JUDGE_API_KEY=<key>
export AETHEREVAL_JUDGE_BASE_URL=https://api.openai.com/v1

aethereval \
  --model /path/to/candidate \
  --model-name candidate \
  --tasks llmeval-med,healthbench,researchqa,arena_hard_v2 \
  --output-dir /output \
  --run-id production-1 \
  --eval-only
```

The model/model-name, output directory, and run id must identify the generation
run. Eval-only automatically inherits its saved generation settings (including
`n`) and rejects conflicting explicit generation overrides. It is intentionally
incompatible with `--overwrite`, and always re-evaluates all existing records
for the selected tasks. Both modes can also be set as `run.generate_only` or
`run.eval_only` in YAML. BFCL maps these flags to its existing generation and
evaluation phases as well.

`safe-alignment` also supports this split. In eval-only mode it starts
Ray-managed SGLang sequence-classification servers over the requested
`dp_size * tp_size` topology and loads only the RM/CM models. RM and CM run
sequentially so each receives the complete GPU budget; the candidate model is
never resident at the same time.

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
  --tasks gpqa-diamond \
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

Benchmark names in CLI, YAML and output JSON use hyphens (for example `gpqa-diamond`).
Benchmark directories also use hyphens; legacy underscore CLI names resolve to
canonical names. Existing output directories are not renamed.

## Reward-Model Metrics

RM-based native tasks can receive reward model paths through shared metric flags:

```bash
aethereval \
  --model /path/to/policy \
  --tasks safe-alignment \
  --output-dir outputs
```

`safe-alignment` defaults to the SGLang-compatible converted checkpoints
`RLLab/Qwen2.5-7B-SafeRLHF-RM` and `RLLab/Qwen2.5-7B-SafeRLHF-CM`; pass
`--rm-model-path` and `--cm-model-path`
only when overriding with local checkpoints. These task-specific defaults live
under `safe-alignment.metrics` in `configs/task_defaults.yaml`.

Optional RM metric flags include `--rm-max-length`, `--rm-dtype`,
`--rm-trust-remote-code`, and repeated `--rm-sglang-arg KEY=VALUE` overrides.

For score-conditioned SFT/RL checkpoints, the separate
[`safe-alignment-dynamic`](benchmarks/safe-alignment-dynamic/README.md) task
sweeps a frozen weight set on the same held-out problems. It reports utility and
paired matching gains, and exports JSON for external reward-curve/cross-utility plotting. Prepare
its HF data once before running; the original `safe-alignment` task is unchanged.

## Open-QA Benchmarks

The following tasks use deterministic short-answer generation (`n=1`, temperature
zero) and local alias-normalized scoring:

- `nq-open` — the 3,610-example public NQ-Open development split; primary normalized
  exact match. The original test labels are not public.
- `triviaqa` — the 11,313-example public `unfiltered.nocontext`
  validation split; primary normalized exact match.

```bash
aethereval \
  --model /path/to/policy \
  --tasks nq-open,triviaqa \
  --output-dir outputs
```

Each task directory includes a preparation script and its exact split/metric notes.

## Native LLM-Judge Benchmarks

These benchmarks use the regular offline backend for candidate generation and an
OpenAI-compatible chat-completions endpoint only for judging:

- `llmeval-med` — 667 items, multi-turn generation, GPT-4o judge, primary `OP`.
- `healthbench` — 5,000 items, GPT-4.1 judge, primary rubric `score`.
- `rar-medical` — official ScaleAI RaR-Medicine test split, Gemma-4 judge,
  primary weighted rubric `score`.
- `rar-science` — official ScaleAI RaR-Science test split, Gemma-4 judge,
  primary weighted rubric `score`.
- `writingbench` — 1,000 items, Claude Sonnet 4.5 judge, primary `overall_score`.
- `creative-writing-v3` — 96 pieces, Claude Sonnet 4.6 judge, primary
  `eqbench_creative_score`.
- `researchqa` — 3,750 items, GPT-4.1-mini judge, primary rubric `coverage`.
- `arena-hard-v2` — 500 hard prompts, GPT-4.1 judge, primary
  `style_controlled_win_rate`.

The aligned per-task judge model and sampling defaults live under each task's
`metrics` section in `configs/task_defaults.yaml`. Judge resolution follows the
same rule as candidate generation: CLI/config values override every selected
task, while omitted values preserve each task's own defaults. Judge settings
remain separate from candidate generation settings.

| Task | Judge temperature | Judge top-p | Judge max new tokens |
| --- | ---: | ---: | ---: |
| `llmeval-med` | 1.0 | 1.0 | 4096 |
| `healthbench` | 0.5 | 1.0 | 2048 |
| `rar-medical` | 1.0 | 1.0 | 4096 |
| `rar-science` | 1.0 | 1.0 | 4096 |
| `writingbench` | 1.0 | 0.95 | 2048 |
| `creative-writing-v3` | 0.0 | 1.0 | 4096 |
| `researchqa` | 0.0 | 1.0 | 4096 |
| `arena-hard-v2` | 0.0 | 1.0 | 16000 |

Upstream LLMEval-Med omits temperature/top-p, and several other upstreams omit
top-p. AetherEval pins those conventional unfiltered values to `1.0` so API and
local judges receive identical sampling settings. OpenAI does not document a
fixed omitted-value token limit, so the otherwise-unspecified LLMEval-Med and
ResearchQA judge limits are pinned to 4096. Both output protocols are far shorter
than this cap.

Online judges use LiteLLM, so OpenAI, Anthropic, Gemini, and other supported
providers share the same benchmark message templates. For native provider routing,
set the provider's normal environment variable and omit `--judge-base-url`:

```bash
export OPENAI_API_KEY=<key>

aethereval \
  --backend sglang \
  --model /path/to/candidate \
  --tasks healthbench \
  --output-dir outputs
```

`--judge-base-url` is reserved for an OpenAI-compatible gateway or custom endpoint;
set `AETHEREVAL_JUDGE_API_KEY=-` for an unauthenticated one. Optional overrides are
`--judge-model`, `--judge-base-url`, `--judge-api-key-env`,
`--judge-workers`, `--judge-timeout`, `--judge-max-retries`, and
`--judge-repeats` (the last one controls LLMEval-Med's three-run protocol).
Judge sampling can be overridden independently with
`--judge-max-new-tokens`, `--judge-temperature`, and `--judge-top-p`.

The same native judge benchmarks can instead load a local judge through
Ray-managed SGLang servers. With judge DP greater than one, AetherEval starts an
SMG router automatically:

```bash
aethereval \
  --model /path/to/candidate \
  --model-name candidate \
  --tasks healthbench \
  --output-dir /output \
  --run-id production-1 \
  --dp-size 8 \
  --tp-size 1 \
  --judge-backend local \
  --judge-model openai/gpt-oss-120b \
  --judge-tp-size 8 \
  --judge-sglang-arg context_length=131072 \
  --judge-sglang-arg mem_fraction_static=0.8
```

If judge DP/TP are omitted, the local judge defaults to one TP replica across
the candidate run's total `dp_size * tp_size` GPU budget. `--judge-dp-size` and
`--judge-tp-size` can override that topology. `--judge-workers` controls the
number of metric workers feeding independent requests into the judge service.
Judge-specific thinking can be set with `--judge-enable-thinking` or
`--no-judge-enable-thinking`. This is applied directly to local judges and sent
as `chat_template_kwargs.enable_thinking` to compatible OpenAI-style judge
endpoints. Omitting both flags defaults an internally managed local judge to
no-thinking; API judging omits the field and preserves the remote provider's
default.

For a normal generate-and-evaluate invocation, local mode automatically runs the
candidate generation phase first, shuts the candidate backend down, and then
loads the judge in eval-only mode. Candidate and judge weights therefore never
occupy GPU memory at the same time. Explicit `--generate-only` and `--eval-only`
commands remain supported as well.

Local judging is opt-in. It preserves each benchmark's existing judge prompt,
sampling settings, and parser, but replacing its official GPT/Claude judge with a
local model changes the evaluation model and the resulting score is not directly
leaderboard-comparable. AetherRL and `/tmp/verl-rubric` also use a local
generative judge (typically GPT-OSS), but their HealthBench reward path batches
all rubrics into a different single prompt, so it is not exactly the official
HealthBench judging protocol used here.

Malformed judge output gets three ordinary format attempts. An internally
managed local SGLang judge then gets one task-specific structured-output attempt
(`json_schema` or `regex`). If that also fails, each benchmark keeps its official
failure behavior rather than applying a shared zero-score fallback: for example,
ResearchQA and Arena-Hard exclude failed judgments, while WritingBench raises and
HealthBench continues retrying. Failure and exclusion counts are included in the
reported metrics where applicable. ResearchQA and Creative Writing failures are
left eligible for scoring again on resume, matching their upstream workflows.

The benchmark folders document the pinned candidate and judge decoding settings.
CLI generation flags override candidate defaults except for the `n>1` sampling
guard described above. Avoid other global decoding overrides when
protocol-aligned scores are required.

## Thinking models

Native tasks support both thinking and non-thinking chat templates:

```bash
# Explicitly enable thinking
aethereval --model Qwen/Qwen3-4B --tasks <task> --enable-thinking

# Explicitly disable thinking
aethereval --model Qwen/Qwen3-4B --tasks <task> --no-enable-thinking
```

Omitting both flags preserves the tokenizer/checkpoint default. In particular,
the original `Qwen/Qwen3-4B` chat template defaults to thinking enabled. The same
setting can be written as `generation.enable_thinking: true` or `false` in YAML.
It is applied locally while rendering the chat template, is shown by `--inspect`,
and is saved in each task's `run_config.json` so `--eval-only` inherits the mode
used by `--generate-only`.

This switch does not automatically change temperature, top-p, output length, or
any task-specific generation defaults. Set those separately only when the target
model and benchmark protocol call for them. BFCL is not affected because its
official adapter builds a ToolRL completion prompt directly instead of applying
the tokenizer chat template.

LiteLLM routes the two Claude-default tasks directly through Anthropic. A second
eval-only command is only necessary when the selected tasks use different API
credentials or gateways:

```bash
export ANTHROPIC_API_KEY=<key>

aethereval \
  --model /path/to/candidate \
  --model-name candidate \
  --tasks writingbench,creative_writing_v3 \
  --output-dir /output \
  --run-id production-1 \
  --eval-only
```

Resuming these tasks preserves completed judge results and judges only newly
generated rows. Use a new `--run-id` or `--overwrite` when changing the judge
model, endpoint behavior, or judging protocol.

## External Benchmarks

Some benchmarks do not fit the native `task.py`/`metrics.py` contract because they own
their own generation loop, agent runtime, or reference output layout. These live under
`benchmarks/<name>/` with an `external.py` API. The CLI task router still lets you
select them with `--tasks`; it dispatches them to their external runner internally.

API-Bank is a native task and should be run with `--tasks apibank`. It keeps `n=1`
and defaults to four complete benchmark repetitions (`num_repeats: 4`), whose numeric
metrics are averaged. Use `--num-repeats 1` for a quick single run.

Current external benchmarks:

- `benchmarks/bfcl` — BFCL V3 wrapper (`bfcl-eval==2025.6.8`):
  `aethereval --tasks bfcl --model <model> --output-dir outputs`

BFCL defaults to `live,non_live,multi_turn` and reports each section's `Acc` and
reference-aware ToolRL-format rate plus `overall_acc` and `overall_format`,
matching common V3 comparison tables. The expected format comes from the BFCL subset and
multi-turn ground truth, so no-tool cases require `<response>` while tool execution
steps require `<tool_call>` and terminate with `<response>`. It runs four independent
repetitions by default (`--num-repeats 1` for a quick single run) and reports their
mean. BFCL keeps `n=1`; `--n` controls completions per prompt, not full benchmark
repetitions. In V3 these three collections together are the full benchmark.
The default `--bfcl-handler toolrl` accepts arbitrary ToolRL-trained checkpoints;
`--bfcl-handler official` instead reuses an exact prompt-mode model registration from
the pinned BFCL package and reports official accuracy metrics without ToolRL format
columns.

External runs use the regular `aethereval` CLI for shared runtime flags
(`--backend`, `--tp-size`, `--gpu-memory-utilization`, `--max-model-len`, etc.) plus
benchmark-specific selectors such as `--categories` for BFCL.

BFCL with SGLang always uses SGLang Model Gateway with cache-aware routing, including
when `--dp-size 1`; `--tp-size` remains the tensor-parallel size per replica. This
avoids the upstream BFCL behavior that treats the total GPU count as tensor
parallelism. BFCL keeps its official generation loop and scorer but connects to
the same Ray-managed SGLang workers and SMG router as native tasks, so its DP
replicas can be placed across an attached multi-node Ray cluster.

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
