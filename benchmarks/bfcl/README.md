# BFCL-v3 (external benchmark)

Reproduces the **Berkeley Function Calling Leaderboard v3** numbers reported by
ToolRL-style papers (`OverallAcc`, `Non-LiveASTAcc`, `Non-LiveExecAcc`, `LiveAcc`,
`MultiTurnAcc`, `RelevanceDetection`, `IrrelevanceDetection`) for ToolRL/GDPO-style
models, i.e. models that emit
`<think>…</think>\n<tool_call>…</tool_call>\n<response>…</response>`.

## Why this is an *external* benchmark

AetherEval's native contract (`task.py` + `metrics.py`) is single-shot: the framework
builds a prompt, generates one response, scores it. BFCL-v3 does not fit that — it owns
its own generation, a **multi-turn agentic execution loop**, and AST/executable
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

BFCL-**v3** data ships in `bfcl-eval==2025.6.8` (later releases moved to v4). The
sglang container pins it (see `docker/Dockerfile.sglang`):

```bash
pip install "bfcl-eval==2025.6.8" --no-deps   # keep the container's sglang/torch stack
```

`--no-deps` skips bfcl's pinned optional API SDKs (cohere/anthropic/…); we only use the
local **sglang** OSS handler, and [`_compat.py`](_compat.py) stubs those unused imports.

## Run

```bash
# base HF model (registry name == HF id):
aethereval --tasks bfcl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs --categories all --num-gpus 1

# a GDPO/GRPO checkpoint (any registry name + a local dir):
aethereval --tasks bfcl \
  --model rlla-gdpo --model-path /path/to/hf_ckpt \
  --output-dir outputs --categories all --num-gpus 1
```

`--categories` accepts BFCL collections/categories: `all`, `non_live`, `live`,
`multi_turn`, or individual ones (`live_simple`, `multi_turn_base`, …). Backend defaults
to **sglang**. If `--num-gpus` is omitted, the AetherEval CLI uses `--dp-size`, then
`--tp-size`, as the BFCL GPU count.

BFCL's local handler sends requests to the local SGLang server concurrently. The upstream
package hardcodes 100 workers, which can overload local servers on long-generation runs;
AetherEval caps that concurrency with `--num-threads` (default: 16). Lower it to `8` or
`4` if you see local `Connection error` messages.

Upstream BFCL also prints very verbose multi-turn step logs (`ID: ..., Turn: ..., Step:
...`, empty-response notices, and separator lines). AetherEval filters those by default;
use `--bfcl-verbose` only when debugging BFCL's internal turn loop.

BFCL is launched via the normal AetherEval task path, so generation/backend settings are
resolved from the same CLI/YAML config stack as native tasks. The SGLang server may still
log the model's default `generation_config` at startup, but the BFCL request overrides the
actual sampling parameters: `temperature`, `top_p`, `top_k`, `max_new_tokens`, and
`context_length` are forwarded where applicable, with `repetition_penalty=1.0`.

### Python API

```python
from pathlib import Path
from benchmarks.bfcl.external import ExternalRunSpec, run

result = run(ExternalRunSpec(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir=Path("outputs/bfcl-base"),
    categories=["all"], backend="sglang", num_gpus=1,
))
print(result.metrics, result.primary_score)   # primary_metric = "OverallAcc"
```

## Output

```text
outputs/<run_id>/bfcl/
  predictions.jsonl   # AetherEval-normalized rows, one BFCL test case per row
  result/   # bfcl raw generations
  score/    # bfcl leaderboard csv (data_overall.csv, data_live.csv, …)
  summary.json   # AetherEval-style: {metrics, primary_metric, primary_score}
```

`predictions.jsonl` follows the standard AetherEval row shape
(`sample_id`, `gen_idx`, `prompt`, `generation`, `score`, `is_pass`, `parsed`,
`gold`, `error`, `meta`). The BFCL raw record is preserved under
`meta.bfcl_record`, and the original `result/` and `score/` trees are retained for
faithful BFCL debugging/re-scoring.

`metrics` keys → ToolRL paper columns:

| metric key              | source (`data_overall.csv`) |
|-------------------------|-----------------------------|
| `OverallAcc`            | `Overall Acc`               |
| `Non-LiveASTAcc`        | `Non-Live AST Acc`          |
| `Non-LiveExecAcc`       | `Non-Live Exec Acc`         |
| `LiveAcc`               | `Live Acc`                  |
| `MultiTurnAcc`          | `Multi Turn Acc`            |
| `RelevanceDetection`    | `Relevance Detection`       |
| `IrrelevanceDetection`  | `Irrelevance Detection`     |
| `correct_format`        | ToolRL format check over generations |

Snake-case aliases (`overall_acc`, `non_live_ast_acc`, etc.) and legacy aliases
(`avg_acc`, `non_live_overall_acc`, `live_overall_acc`, `multi_turn_overall_acc`)
are also emitted for compatibility.

GDPO Table 1 reference (avg of 5 runs) for sanity-checking:

```
Qwen2.5-Instruct-1.5B  Live 37.89  Multi 0.12  Non-Live 15.63  Avg 17.88  Format 4.74
  + GRPO               Live 50.63  Multi 2.04  Non-Live 37.87  Avg 30.18  Format 76.33
  + GDPO               Live 55.36  Multi 2.50  Non-Live 40.58  Avg 32.81  Format 80.66
```

## Files

- `handler.py` — `RLLAHandler` (bfcl `OSSHandler`): faithful ToolRL
  `benchmarks/BFCL/rlla_qwen.py` prompt + `<tool_call>` decode, adapted to `bfcl_eval`.
- `register.py` — inject the handler into bfcl's `MODEL_CONFIG_MAPPING`.
- `external.py` — `ExternalRunSpec` / `ExternalResult` / `run()` + score parsing.
- `_compat.py` — stub drifted optional API SDKs so `bfcl_eval` imports under `--no-deps`.

> Multi-turn note: BFCL's new handler API drops the per-call `turn_type`; the handler
> infers multi-turn from tool-feedback in the message history. Single-turn (Non-Live /
> Live, which dominate the overall report) is exact; multi-turn uses the same ToolRL
> template.
