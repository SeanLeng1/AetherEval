# BFCL-v4 (external benchmark)

Runs the official **Berkeley Function Calling Leaderboard v4** generation loop and
scorer for ToolRL/GDPO-style models, i.e. models that emit
`<think>…</think>\n<tool_call>…</tool_call>\n<response>…</response>`.

## Why this is an *external* benchmark

AetherEval's native contract (`task.py` + `metrics.py`) is single-shot: the framework
builds a prompt, generates one response, scores it. BFCL-v4 does not fit that — it owns
its own generation, **multi-turn and agentic execution loops**, and AST/executable
checkers. Faithfully reproducing it requires the official `bfcl_eval` package (its test
data, AST checker, and multi-turn runtime), not a re-implementation.

So BFCL is wrapped here as an **external benchmark**: a self-contained extension under
`benchmarks/bfcl/` with an explicit `run()` API. It does not provide native
`task.py`/`metrics.py` files, but the AetherEval CLI task router still accepts
`--tasks bfcl` and dispatches to this external runner. It drives `bfcl_eval`
in-process while writing an AetherEval-style `summary.json`. Any future external
benchmark can follow the same
`ExternalRunSpec` / `ExternalResult` / `run` shape.

## Requirements

The production SGLang image pins the official evaluation release
`bfcl-eval==2025.12.17` in AetherRL's `docker/Dockerfile.sglang`. Its local parser and
agentic dependencies are installed explicitly without replacing the image's
SGLang/torch stack. The legacy Dockerfile in this repository is not the production
image build source.

```bash
pip install "bfcl-eval==2025.12.17" --no-deps
```

[`_compat.py`](_compat.py) only stubs unused model-provider SDK imports. Real
Tree-sitter parsers are mandatory so Java/JavaScript scores cannot silently degrade.
The image does not pre-cache BFCL's `all-MiniLM-L6-v2` memory-vector encoder. Selecting
`memory`, `agentic`, or `all` emits a warning and requires either a populated Hugging
Face cache or network access. The default comparison-table categories never load it.

The default selection is `live,non_live,multi_turn`, matching comparison tables that
report those three sections and their macro average. It is fully offline after the BFCL
data and model are cached. An official full V4 run (`--categories all`) also includes
live web search and therefore requires both outbound network access and
`SERPAPI_API_KEY`. AetherEval fails before generation when a selected collection includes
`web_search` but the key is absent.

BFCL's aligned generation defaults (`temperature`, `max_new_tokens`, `top_p`, and
`top_k`) live under `bfcl` in `configs/task_defaults.yaml`, alongside native task
defaults. Global generation CLI/YAML values override them in the same way as other
tasks.

BFCL also defaults to `n: 4`, matching ToolRL tables reported as the mean of four
evaluation runs. Each run performs independent generation and official scoring with
seeds `0,1,2,3` by default; an explicit `--seed S` uses `S,S+1,S+2,S+3`. The SGLang
service stays loaded across all four runs. Use `--n 1` for a single diagnostic run.
The official BFCL V4 generation default remains `temperature=0.001`, which AetherEval
keeps for faithful comparison.

## Run

```bash
# base HF model (registry name == HF id):
aethereval --tasks bfcl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs --dp-size 1 --tp-size 1

# a GDPO/GRPO checkpoint (any registry name + a local dir):
aethereval --tasks bfcl \
  --model /path/to/hf_ckpt \
  --model-name optional-output-label \
  --output-dir outputs --dp-size 1 --tp-size 1
```

`--model` is always the actual Hugging Face ID or local checkpoint used for loading.
Optional `--model-name` only controls the shared AetherEval output label, including for
IDs such as `qwen2.5/huggingface`; it does not alter BFCL model registration or serving.

`--categories` accepts V4 collections/categories: `all`, `all_scoring`, `non_live`,
`live`, `multi_turn`, `agentic`, `memory`, `web_search`, `format_sensitivity`, or
individual ones (`simple_python`, `memory_vector`, …). Backend defaults to **sglang**.
Omitting it selects `live,non_live,multi_turn`; pass `--categories all` only when the
official V4 agentic/web-search aggregate is required.
BFCL inherits AetherEval's normal parallelism semantics: `--dp-size` is
the replica count and `--tp-size` is the tensor-parallel size of each replica.
AetherEval replaces BFCL's upstream TP-only launch command with SGLang Model Gateway
(SMG), including for one replica, using `cache_aware` routing by default:

```bash
aethereval --tasks bfcl --model /path/to/hf_ckpt \
  --backend sglang --dp-size 8 --tp-size 1
```

Use `--bfcl-router-policy` to change the policy.

BFCL's local handler sends requests to the local SGLang server concurrently. The upstream
package hardcodes 100 workers. AetherEval defaults to `max(16, 16 * dp_size)`, capped at
100, so every replica receives useful batching pressure. Override with `--num-threads`;
lower it if server logs show frequent KV-cache retractions or connection errors.
Deterministic context-length rejections are never retried, even when SMG surfaces a
worker rejection as HTTP 500. Other transient connection, 429, and 5xx failures retain
the bounded retry path. A single native generation request has a 30-minute read timeout,
preventing a dead router request from silently holding the benchmark for many hours.

Upstream BFCL also prints very verbose multi-turn step logs (`ID: ..., Turn: ..., Step:
...`, empty-response notices, and separator lines). AetherEval filters those by default;
use `--bfcl-verbose` only when debugging BFCL's internal turn loop.

Generation resumes from existing BFCL JSONL files unless `--overwrite` is set. If a
previous process was killed while writing the final record, AetherEval preserves the
incomplete bytes in a neighboring `*.corrupt-tail` file, removes only that incomplete
record, and lets BFCL regenerate its test case. Corruption before the final record is
reported with the exact file and line and is never repaired automatically.

BFCL is launched via the normal AetherEval task path, so generation/backend settings are
resolved from the same CLI/YAML config stack as native tasks. The SGLang server may still
log the model's default `generation_config` at startup, but the BFCL request overrides the
actual sampling parameters: `temperature`, `top_p`, `top_k`, `max_new_tokens`, and
`context_length` are forwarded where applicable, with `repetition_penalty=1.0`.
An explicit `--seed` is forwarded too, for reproducible local sampling.
`--mem-fraction-static`, `--dtype`, and compatible repeatable `--sglang-arg` values are
also forwarded to the gRPC SGLang workers. Tokenizer, chat-template, reasoning-parser,
and tool-parser settings are mirrored onto SMG because gRPC mode runs those components
in the router.
SGLang worker logging defaults to `error` and SMG router logging to `warn`. Use
`--bfcl-sglang-arg router_log_level=info` when detailed router diagnostics are needed.

BFCL runs after native tasks and starts its own server. Use `--bfcl-context-length` and
repeatable `--bfcl-sglang-arg` values when BFCL needs a server setting that must not
affect APIBank or other native tasks. For example, an explicit static-YaRN experiment is:

```bash
aethereval --tasks apibank,bfcl --backend sglang --dp-size 8 --tp-size 1 \
  --bfcl-context-length 131072 \
  --bfcl-sglang-arg \
  'json_model_override_args={"max_position_embeddings":131072,"rope_parameters":{"rope_theta":1000000.0,"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
```

The checkpoint uses the Transformers-v5 `rope_parameters` schema. Keep its original
`rope_theta=1000000.0` in the override; passing the older `rope_scaling` object replaces
`rope_parameters` without `rope_theta` and makes current SGLang fail during model startup.

This override is never enabled automatically: static YaRN changes position encoding for
short inputs too, and a tokenizer's `model_max_length` does not establish that the model
was trained or validated at that length. Compare benchmark accuracy before adopting it.

### Python API

```python
from pathlib import Path
from benchmarks.bfcl.external import ExternalRunSpec, run

result = run(ExternalRunSpec(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir=Path("outputs/bfcl-base"),
    backend="sglang", dp_size=1, tp_size=1,
))
print(result.metrics, result.primary_score)   # primary_metric = "avg_acc"
```

## Output

```text
outputs/<run_id>/bfcl/
  predictions.jsonl   # all runs; gen_idx identifies the evaluation run
  result/run_01/ ... result/run_04/   # BFCL raw generations
  score/run_01/  ... score/run_04/    # official per-run leaderboard CSVs
  summary.json   # averaged metrics plus per-run metrics under runs[]
```

`predictions.jsonl` follows the standard AetherEval row shape
(`sample_id`, `gen_idx`, `prompt`, `generation`, `score`, `is_pass`, `parsed`,
`gold`, `error`, `meta`). The BFCL raw record is preserved under
`meta.bfcl_record`, and the original `result/` and `score/` trees are retained for
faithful BFCL debugging/re-scoring.

Canonical `metrics` keys → official V4 CSV columns or AetherEval aggregation:

| metric key              | source (`data_overall.csv`) |
|-------------------------|-----------------------------|
| `official_overall_acc`  | `Overall Acc`               |
| `non_live_acc`          | `Non-Live AST Acc`          |
| `live_acc`              | `Live Acc`                  |
| `multi_turn_acc`        | `Multi Turn Acc`            |
| `agentic_acc`           | `data_agentic.csv`: `Agentic Overall Acc` |
| `web_search_acc`        | `Web Search Acc`            |
| `memory_acc`            | `Memory Acc`                |
| `relevance_detection`   | `Relevance Detection`       |
| `irrelevance_detection` | `Irrelevance Detection`     |
| `format_sensitivity_max_delta` | `Format Sensitivity Max Delta` |
| `format_sensitivity_std` | `Format Sensitivity Standard Deviation` |
| `live_format`           | reference-aware ToolRL format rate over Live completions |
| `non_live_format`       | reference-aware ToolRL format rate over Non-Live completions |
| `multi_turn_format`     | reference-aware ToolRL format rate over Multi-Turn completions |
| `avg_acc`               | macro mean of Live/Non-Live/Multi-Turn Acc |
| `avg_format`            | macro mean of Live/Non-Live/Multi-Turn Format |

`avg_acc = (live_acc + non_live_acc + multi_turn_acc) / 3`, and the Format average
uses the same unweighted three-way mean. This exactly matches the `Average` columns in
Live/Non-Live/Multi-Turn comparison tables; it is intentionally different from BFCL
V4's `official_overall_acc`, which additionally includes the Agentic benchmark with the
leaderboard's weighting. No duplicate CamelCase or legacy aliases are emitted.
Every displayed metric is first averaged across the four independent runs; the final
`avg_acc` and `avg_format` are then recomputed from the three averaged section
columns to avoid rounding drift.

Format is an AetherEval comparison metric, not an official BFCL leaderboard column.
It follows ToolRL's public regex-and-tag-count reward while deriving the expected shape
from BFCL: ordinary function-calling subsets require `<tool_call>`, irrelevance subsets
require `<response>`, and each multi-turn step is classified from that turn's official
ground truth. A tool-requiring turn uses `<tool_call>` for execution steps and must end
with `<response>`; `multi_turn_miss_func` and `multi_turn_miss_param` turns whose ground
truth is empty require `<response>`. The percentage is computed over actual model
completions, which is the same unit scored by ToolRL's format reward.

## Files

- `handler.py` — `RLLAHandler` (bfcl `OSSHandler`): faithful ToolRL
  `benchmarks/BFCL/rlla_qwen.py` prompt + `<tool_call>` decode, adapted to `bfcl_eval`.
- `register.py` — inject the handler into bfcl's `MODEL_CONFIG_MAPPING`.
- `external.py` — `ExternalRunSpec` / `ExternalResult` / `run()` + score parsing.
- `_compat.py` — isolate unused provider SDK imports without stubbing scoring parsers.

> Multi-turn note: BFCL's handler API drops the per-call `turn_type`; the handler
> infers multi-turn from tool-feedback in the message history. Single-turn,
> multi-turn, and agentic requests use the trained ToolRL template; V4 memory-agent
> system instructions are preserved inside that template.
