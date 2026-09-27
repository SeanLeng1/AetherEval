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

Inside the AetherRL Docker image, use its active Python environment and preserve
the preinstalled inference/scoring dependencies:

```bash
python -m pip install --no-deps -e .
```

Outside that image, install the core dependencies with `python -m pip install -e .`
and provision the chosen inference backend separately. The SGLang backend (candidate
generation, local judges and reward models) needs SGLang >= 0.5.18 and
smg-grpc-servicer >= 0.8.0, plus sglang-router for the SMG router; the AetherRL
sglang0.5.20 image ships SGLang 0.5.20, smg-grpc-servicer 0.9.1 and sglang-router
0.3.2. Each worker checks the floor at startup and names any package that is
missing or too old. HumanEval+ and MBPP+ also
require the scoring-only EvalPlus installation (already included in AetherRL Docker):

```bash
python -m pip install --no-deps \
  "evalplus @ git+https://github.com/evalplus/evalplus.git@26d6d00bb1fd0fa37f39c99d5290da67891d1c5e" \
  tempdir==0.7.1 wget==3.2 appdirs==1.4.4 termcolor==3.3.0
```

EvalPlus is pinned to upstream commit `26d6d00`, which includes the HumanEval/32
`find_zero` fix. No local oracle patch or patch command is needed. This revision
also uses upstream's 4-second minimum per-test time limit (0.3.1 used 1 second).
After upgrading an existing environment, restart evaluation and re-score saved
HumanEval+/MBPP+ generations with `--eval-only`; regeneration is unnecessary.

The runtime must also provide NumPy, psutil and tqdm. MBPP+ additionally uses
`tree-sitter-python` and EvalPlus's official sanitizer. AetherRL Docker pins
`tree-sitter==0.21.3` for BFCL, installs `tree-sitter-python==0.21.0`, and adapts
the sanitizer's parser constructor to that API. It checks both MBPP+ and
BFCL parsers at image build time. Test execution and sanitizing rules otherwise
come directly from the pinned upstream revision.
For a separate environment without the BFCL pin, EvalPlus's native sanitizer
instead requires `tree-sitter>=0.22.0` with a compatible Python grammar package.

To repair an older AetherRL container without rebuilding, run the scoring-dependency
command above, then apply the same parser-constructor adaptation:

```bash
python -m pip install --no-deps tree-sitter-python==0.21.0
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import evalplus

assert version("tree-sitter") == "0.21.3"
path = Path(evalplus.__file__).with_name("sanitize.py")
old = "parser = Parser(Language(tree_sitter_python.language()))"
new = 'parser = Parser()\n    parser.set_language(Language(tree_sitter_python.language(), "python"))'
source = path.read_text()
assert old in source or new in source, "Unexpected EvalPlus sanitizer version"
path.write_text(source.replace(old, new))
from evalplus.data.mbpp import mbpp_deserialize_inputs
from evalplus.sanitize import sanitize
assert sanitize("def f(x):\n    return x", entrypoint="f") == "def f(x):\n    return x"
print("MBPP+ imports and sanitizer OK")
PY
```

EvalPlus is intentionally not an automatic project dependency: its unused Gemini
provider dependencies require protobuf `<6`, conflicting with SMG's generated
protobuf code, and its tree-sitter requirements replace the Docker's BFCL pins.
Installing EvalPlus with `--no-deps` once does not prevent a later dependency-resolving
install from pulling those dependencies. If the environment was already changed,
start a fresh container from the original image before using the command above;
editing this project's dependencies does not restore downgraded packages.

## List tasks

```bash
aethereval --list-tasks
```

```bash
aethereval --list-task-defaults
```

Task generation defaults are centrally defined in `configs/task_defaults.yaml`.
You can edit this file to adjust per-task `n/max_new_tokens/temperature/top_p`.
Each benchmark README has an **Official source and protocol** section with the
audited repository/release, reference settings and known differences. Some datasets
do not define a universal decoder: their defaults are explicitly labeled local
profiles, not invented official standards. Token limits count new output tokens,
not prompt plus output; larger reasoning budgets must be declared as overrides.
Matching sampling alone does not imply matching prompts, data slices or graders.
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

### CPU scoring parallelism

Add `--num-proc 32` (or `metrics.num_proc: 32` in YAML) to parallelize local
per-response scoring, including math-verify, HumanEval+, MBPP+ and LiveCodeBench.
The default is `1`. This also works with `--eval-only` and resume; it does not
regenerate answers or change grading, aggregation, or the output order.
Choose the count for the CPU cores and memory available on the invoking node,
not the GPU count. Code workers can launch their existing test subprocesses.
HumanEval+ and MBPP+ always score in spawned workers (one at the default) with
BLAS threads capped at 1 unless already set in the environment, so EvalPlus's
4 GiB candidate memory limit does not depend on the node's core count. Each check
then starts a fresh interpreter, so give these tasks a larger `--num-proc`.
Earlier versions could score them lower on many-core nodes; re-score those
checkpoints with `--eval-only` before comparing.
RM batch scoring and LLM-judge concurrency (`--judge-workers`) are unchanged.

### Ray data parallelism

Evaluation commands run directly in the invoking shell. SGLang data-parallel
replicas are Ray actors behind one SMG router:

```bash
aethereval --model /path/to/model --tasks ifeval --dp-size 8 --tp-size 1
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

AetherEval keeps at most 64 SGLang requests in flight per replica (`64 x dp-size`
in total). `--sglang-arg inflight_per_replica=128` (or the same key in
`sglang.extra_model_kwargs` and `--rm-sglang-arg`) raises the cap for candidate
generation and reward-model scoring; the local judge's concurrency is set by
`--judge-workers` and BFCL's by `--num-threads`, which this cap does not limit.
It helps only when the KV cache has room for more sequences, for example
models with few KV heads or short-output tasks on 80 GB+ GPUs; Qwen3-4B/8B at a
16k budget already come close to the KV limit at 64. Compare generation time on
one checkpoint before keeping a larger value. A different cap changes SGLang's
batching, so individual sampled outputs differ, as they already do between runs.

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

When a normal run resumes, non-judge metrics rescore every record. LLM-judge
tasks judge only new or unscored records and keep stored judgments whose judge
settings are unchanged: each judged record stores a fingerprint of the
`judge_*` options (model, backend, endpoint, temperature, top-p, max tokens,
repeats, thinking and local SGLang arguments, but not workers, timeouts,
retries, API-key variable or judge DP/TP), and records with a different or
missing fingerprint are judged again. A rerun whose judgments are all reused
does not start the local judge. The generation phase leaves the `summary.json`
of an already evaluated task untouched when it has nothing to generate. Use
`--eval-only` to re-judge every record, for example after changing a judge
prompt or parser, or a judge endpoint set only through environment variables.
Each evaluation also records the installed scorer versions (math-verify, which
`pyproject.toml` pins to the validated 0.9.0, latex2sympy2_extended, EvalPlus,
NLTK, langdetect, SGLang, vLLM) as `scoring_packages` in the task's
`run_config.json`; re-score saved generations with `--eval-only` after
changing them.

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
incompatible with `--overwrite`, and always re-evaluates (and re-judges) all
existing records for the selected tasks. Both modes can also be set as
`run.generate_only` or `run.eval_only` in YAML. BFCL maps these flags to its
existing generation and evaluation phases as well.

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

## Rebuild Benchmark Data

From the repository root, after installing preparation dependencies (`pip install -e '.[dynamic]'`):

```bash
python benchmarks/init.py                     # all native benchmarks
python benchmarks/init.py minerva aime25      # selected benchmarks only
```

The entry point calls each benchmark's existing `prepare_data.py` in a separate
process and stops on the first failure. It overwrites prepared files in
`benchmarks/<task>/data/`; it does not run generation, judging, or modify evaluation
results. Source downloads may use their normal caches. BFCL is excluded because
its official external runner manages its data. No temporary upstream checkout is
required. Task-specific options remain available via
`python -m benchmarks.<task>.prepare_data --help` where supported.

Most sources are pinned to a commit, release or dataset revision (the GPQA CSV by
sha256). The exceptions: Arena-Hard baseline answers and its style cohort follow
the upstream `main` branch, HealthBench reads a fixed but unhashed blob URL, and
ZebraLogic reproduces its committed file only with gated access to
`allenai/ZebraLogicBench-private` (the public fallbacks record a different
`source_dataset`).

Dynamic Safe Alignment also regenerates `data/protocol.json` from current training
score statistics by default. To preserve a particular RL/SFT experiment's mapping,
prepare that task with `--rl-data` or `--revision` instead. Rebuilt prompts or gold
answers may differ when an upstream dataset has changed; old evaluation results
are not automatically refreshed. The fingerprint test (see [Development](#development))
names every task whose rows, prompts, golds or scoring metadata a rebuild changed.

## Benchmark Contract

Each benchmark folder must include:

```text
benchmarks/<task_name>/
  README.md
  data/*.jsonl
  task.py
  metrics.py
  prepare_data.py
  prompts/          # optional static prompt templates / criteria
```

`task.py` must define:

- `TASK_NAME: str`
- `DATA_FILE: str` (must be `.jsonl`)
- `load_samples(task_dir) -> list[Sample]`
- `build_prompt(sample) -> str | list[dict]`

Per-task generation defaults are loaded only from `configs/task_defaults.yaml`; a
`task.py` that defines `DEFAULT_GEN` is rejected. Loading a task fails on any
generation key the backends do not understand (`n`, `max_new_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `stop`, `seed`, `enable_thinking`,
`regex`, `json_schema`, `ebnf`, `structural_tag`), so a typo cannot silently fall
back to a default.

Prompt handling:

- Framework defaults to chat-format generation.
- If `build_prompt` returns `str`, it is auto-wrapped to `[{"role":"user","content": ...}]`.
- Offline backends render prompts with tokenizer `apply_chat_template`; if unavailable, the framework falls back to plain `role: content` text and prints a warning.

`metrics.py` must define `aggregate` and at least one scoring entry point:

- `score_generation(sample, generation) -> dict` (`score` required)
- or `score_generations_batch(samples, generation_outputs, metric_options) -> list[list[dict]]`
- `aggregate(sample_results, metric_options) -> dict[str, float]`

Recommended:

- `PRIMARY_METRIC: str` (used by runner to surface report metric in `summary.json`)
Batch scoring takes precedence when both entry points exist. Batch-only RM and
judge tasks do not need a placeholder `score_generation`. The returned outer list
must align with `generation_outputs`; each inner list aligns with its `generations`.

Reuse shared functions through explicit imports when no task-specific logic is
needed; do not add forwarding wrappers or a new task base class. Prompt templates
and grading rules remain benchmark-specific. `prepare_data.py` only builds the
offline data; generation settings belong in `configs/task_defaults.yaml`, sorted
in the same order as benchmark directories. BFCL remains an external adapter for
its official multi-turn execution loop, not a single-response scorer.

A new benchmark also needs, each checked by the test suite:

- an entry in `configs/task_defaults.yaml` (`n>1` only with `temperature>0`);
- an **Official source and protocol** README section that states the configured
  `n=` and `max_new_tokens=` values;
- a fingerprint pin in `tests/golden/benchmark_fingerprints.json` (see
  [Development](#development)), plus a scoring unit test in `tests/`.

Shared benchmark implementation code lives in `benchmark_utils/`, outside
`benchmarks/`, so helper modules are not visually or programmatically mixed with
task folders.

Benchmark names in CLI, YAML and output JSON use hyphens (for example `gpqa-diamond`).
Benchmark directories also use hyphens; legacy underscore CLI names resolve to
canonical names. Existing output directories are not renamed.

### Optional Hooks

The runner also looks for these optional names:

| Module | Hook | Effect |
| --- | --- | --- |
| `task.py` | `load_protocol(task_dir) -> dict` | Saved as `protocol` in `run_config.json` and passed to metrics as `_protocol`; resuming with a different protocol is rejected. |
| `task.py` | `generate_outputs(backend, samples, pending_indices, existing_records, gen_cfg)` | Replaces the single `backend.generate` call, for multi-turn or retried generation. Returns `list[GenerationOutput]`. |
| `metrics.py` | `USES_LLM_JUDGE = True` | With `judge_backend: local`, a shared local judge is passed as `_judge_client`. |
| `metrics.py` | `PRESERVE_EXISTING_SCORES_ON_RESUME = True` | On a normal rerun, records judged with unchanged judge settings keep their scores instead of being judged again; `--eval-only` re-judges them (see the resume notes above). |
| `metrics.py` | `REQUIRES_BACKEND = True` | Scoring receives a backend as `_backend`: one passed to `run_evaluation(backend=...)`, otherwise the one `create_evaluation_backend(metric_options, dp_size, tensor_parallel_size)` returns, which CLI runs always use. |
| `metrics.py` | `validate_metric_options(options)` | Called with the task's metric options plus `n` before a phase that scores; raise to reject the run early. |

Metric options whose names start with `_` are runtime objects and are not written
to `run_config.json`.

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

## Native LLM-Judge Benchmarks

These benchmarks use the regular offline backend for candidate generation and an
OpenAI-compatible chat-completions endpoint only for judging:

- `llmeval-med` — 667 items, multi-turn generation, GPT-4o judge, primary `OP`.
- `healthbench` — 5,000 items, GPT-4.1 judge, primary rubric `score`.
- `writingbench` — 1,000 items, Claude Sonnet 4.5 judge, primary `overall_score`.
- `creative-writing-v3` — 96 pieces, Claude Sonnet 4.6 judge, primary
  `eqbench_creative_score`.
- `researchqa` — 3,750 items, GPT-4.1-mini judge, primary rubric `coverage`.
- `arena-hard-v2` — 500 hard prompts + 250 creative-writing prompts, GPT-4.1 judge,
  primary `style_controlled_win_rate` (hard prompts); `creative_writing_win_rate`
  is reported separately.

The documented per-task judge model and sampling defaults (`judge_model`,
`judge_temperature`, `judge_top_p`, `judge_max_new_tokens`) live under each task's
`metrics` section in `configs/task_defaults.yaml`. Judge resolution follows the
same rule as candidate generation: CLI/config values override every selected
task, while omitted values preserve each task's own defaults. Judge settings
remain separate from candidate generation settings.

Upstream LLMEval-Med omits temperature/top-p, and several other upstreams omit
top-p. AetherEval pins those conventional unfiltered values to `1.0` so API and
local judges receive explicit sampling settings; this does not prove parity with
an upstream model's inherited defaults. OpenAI does not document a
fixed omitted-value token limit, so the otherwise-unspecified LLMEval-Med and
ResearchQA judge limits are pinned to 4096. These are local caps, not official
token-limit requirements or guarantees against truncation.
Where `judge_top_p` is omitted (Creative Writing v3), API judges send no top-p and
a local judge uses `1.0`. A Claude judge reached through LiteLLM's native Anthropic
route (no `--judge-base-url`) never receives top-p together with a temperature,
because that API rejects the combination; WritingBench's `0.95` is not sent there.

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

During eval-only, non-judge metrics run first, then local judge tasks are grouped
by model and runtime configuration (DP/TP and SGLang arguments).
Each group shares one judge, loaded by the group's first task that has records
to judge (so a group whose judgments are all reused never loads it) and unloaded
before the next group.
Per-task prompts, sampling settings and scoring rules remain separate, and API
judging and candidate generation retain their original task order.

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
HealthBench keeps re-sampling like simple-evals, for at most 50 extra attempts,
and then raises. Failure and exclusion counts are included in the
reported metrics where applicable. On a normal rerun, ResearchQA and Arena-Hard
keep their excluded judgments like any other stored judgment, while Creative
Writing failures are judged again, matching their upstream workflows. A Creative
Writing failure is reported as a warning and keeps the task's
`evaluation_complete` false instead of aborting the evaluation phase.

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

Before scoring or judging, graders see the text after the last `</think>`, regardless
of whether `<think>` is present. Without a closing tag, the response is preserved;
an empty suffix after a closing tag remains empty. This text-level rule also treats
literal closing tags in generated code as boundaries. `predictions.jsonl` keeps the
raw generation.

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

A normal rerun of these tasks keeps completed judgments made with the same judge
settings and judges only new or unscored rows; this explicit `--eval-only`
command re-judges every row. Use `--eval-only` after changing the judging
protocol, and a new `--run-id` to keep both results.

## External Benchmarks

Some benchmarks own their generation loop, agent runtime, or reference output layout
and do not fit the native `task.py`/`metrics.py` contract. They live under
`benchmarks/<name>/` with an `external.py` API (`run(spec) -> ExternalResult`), are
still selected with `--tasks`, share the regular runtime flags (`--backend`,
`--dp-size`, `--tp-size`, `--context-length`, ...), and write an AetherEval-style
`summary.json` next to their reference-format raw outputs.

The only external benchmark is BFCL V3 (`bfcl-eval==2025.6.8`):
`aethereval --tasks bfcl --model <model> --output-dir outputs`. It runs one complete
benchmark pass by default (`num_repeats: 1` in `configs/task_defaults.yaml`; pass
`--num-repeats 4` for a repeated-run mean) and reports per-section accuracy and
ToolRL-format rates plus `overall_acc`. See
[benchmarks/bfcl/README.md](benchmarks/bfcl/README.md) for handlers, categories,
SGLang routing, metrics and output layout.

The CLI registers each external benchmark in `aethereval/cli.py` `EXTERNAL_TASKS`;
its `cli.py` provides `add_arguments(parser)` and
`run_external(args, resolved, output_dir)`, which returns the task's `summary.json`
contents.

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

Per run (`<run-id>` only when `--run-id` is given):

```text
<output-dir>/<model-name>[/<run-id>]/
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
- `meta` (includes `prompt_token_count`, `response_token_count`, and
  `finish_reason` for new generations)

`summary.json` is task-level aggregate, and includes:

- `metrics`: full metric dict from task aggregate
- `metrics.avg_prompt_tokens` / `metrics.avg_response_tokens`: model-tokenized
  average prompt and response lengths
- `token_usage`: average and total prompt/response token counts
- `metrics.avg_completed_response_tokens`: mean response tokens only for normal
  EOS / configured-stop endings (`finish_reason="stop"`), regardless of correctness.
  Length-truncated, aborted, and unknown-ending responses are excluded from this
  additional length metric only; scores and the original length metric are unchanged.
  `token_usage` also records this mean, `num_completed_responses`, and
  `total_completed_response_tokens`. With no known completed responses, the mean
  is `null` in `token_usage` and omitted from `metrics`, not reported as zero.
  Legacy predictions without finish reasons remain unknown on resume/rescoring.
  Repeats pool completed responses; the run summary retains task-macro averaging.
- `primary_metric`: report metric name
- `primary_score`: report metric value

`run_summary.json` is run-level summary:

- `results`: all per-task summaries in the run directory, native and external,
  including tasks from earlier invocations
- `phase`: the phase this invocation requested (`generate_and_eval` for a normal run)
- `primary_scores`: each task's primary metric name/value
- `primary_score_aggregate`: mean of task `primary_score` values (direct average across tasks)
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

## Development

Run the checks from the repository root. Outside the AetherRL image, install the
tools with `python -m pip install -e '.[test]'`; inside it, install only `pytest`
and `ruff` so the pinned runtime stays untouched.

```bash
ruff check .
python -m pytest -q        # or: python -m unittest discover -s tests
```

- Tests that need an optional scorer or runtime (EvalPlus, the BFCL V3 stack) are
  skipped with a reason when it is missing. Set `AETHEREVAL_REQUIRE_ALL_TEST_DEPS=1`
  in the runtime image to make them fail instead.
- Data tests read the benchmark files, so fetch the Git LFS data first.
- `tests/test_sglang_installed.py` runs only where SGLang and smg-grpc-servicer are
  installed. It needs no GPU and checks the SGLang/SMG internals the worker patches
  and launch flags rely on, so run it in the image after every SGLang or SMG upgrade.
- `tests/test_benchmark_fingerprints.py` pins each native task's row count and the
  hashes of its rendered prompts and of its golds and scoring metadata, and checks
  that math, MCQ and BBH golds pass their own scorers. After a deliberate data or
  prompt change, refresh the pins and review the diff:
  `AETHEREVAL_UPDATE_GOLDEN=1 python -m unittest tests.test_benchmark_fingerprints`.
  `AETHEREVAL_SLOW_TESTS=1` also runs every HumanEval+/MBPP+ canonical solution.
- The vendored IFEval/IFBench checkers are compared with their pinned upstream
  commits by `tools/check_vendored.py`; see each library's `NOTICE`.
