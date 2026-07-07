# BFCL-v3 (external benchmark)

Reproduces the **Berkeley Function Calling Leaderboard v3** numbers in GDPO Table 1
(Live / Non-Live / Multi-Turn Overall Acc, Avg Acc, Correct Format) for ToolRL/GDPO-style
models — i.e. models that emit
`<think>…</think>\n<tool_call>…</tool_call>\n<response>…</response>`.

## Why this is an *external* benchmark

AetherEval's native contract (`task.py` + `metrics.py`) is single-shot: the framework
builds a prompt, generates one response, scores it. BFCL-v3 does not fit that — it owns
its own generation, a **multi-turn agentic execution loop**, and AST/executable
checkers. Faithfully reproducing it requires the official `bfcl_eval` package (its test
data, AST checker, and multi-turn runtime), not a re-implementation.

So BFCL is wrapped here as an **external benchmark**: a self-contained extension under
`benchmarks/bfcl/` with an explicit `run()` API. AetherEval's task discovery ignores it
(no `task.py`/`metrics.py`), and it drives `bfcl_eval` in-process while writing an
AetherEval-style `summary.json`. Any future external benchmark can follow the same
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
python -m benchmarks.bfcl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/bfcl-base --categories all --num-gpus 1

# a GDPO/GRPO checkpoint (any registry name + a local dir):
python -m benchmarks.bfcl \
  --model rlla-gdpo --model-path /path/to/hf_ckpt \
  --output-dir outputs/bfcl-gdpo --categories all --num-gpus 1
```

`--categories` accepts BFCL collections/categories: `all`, `non_live`, `live`,
`multi_turn`, or individual ones (`live_simple`, `multi_turn_base`, …). Backend defaults
to **sglang**.

### Python API

```python
from pathlib import Path
from benchmarks.bfcl.external import ExternalRunSpec, run

result = run(ExternalRunSpec(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir=Path("outputs/bfcl-base"),
    categories=["all"], backend="sglang", num_gpus=1,
))
print(result.metrics, result.primary_score)   # primary_metric = "avg_acc"
```

## Output

```text
outputs/<run>/
  result/   # bfcl raw generations
  score/    # bfcl leaderboard csv (data_overall.csv, data_live.csv, …)
  summary.json   # AetherEval-style: {metrics, primary_metric, primary_score}
```

`metrics` keys → GDPO Table 1 columns:

| metric key                | Table 1 column        | source (data_overall.csv) |
|---------------------------|-----------------------|---------------------------|
| `non_live_overall_acc`    | Non-Live Overall Acc  | `Non-Live AST Acc`        |
| `live_overall_acc`        | Live Overall Acc      | `Live Acc`                |
| `multi_turn_overall_acc`  | Multi Turn Overall Acc| `Multi Turn Acc`          |
| `avg_acc`                 | Avg Acc               | mean of the three overalls |
| `correct_format`          | Correct Format        | ToolRL format check over generations |

GDPO Table 1 reference (avg of 5 runs) for sanity-checking:

```
Qwen2.5-Instruct-1.5B  Live 37.89  Multi 0.12  Non-Live 15.63  Avg 17.88  Format 4.74
  + GRPO               Live 50.63  Multi 2.04  Non-Live 37.87  Avg 30.18  Format 76.33
  + GDPO               Live 55.36  Multi 2.50  Non-Live 40.58  Avg 32.81  Format 80.66
```

## Files

- `handler.py` — `RLLAHandler` (bfcl `OSSHandler`): ToolRL prompt + `<tool_call>` decode.
- `register.py` — inject the handler into bfcl's `MODEL_CONFIG_MAPPING`.
- `external.py` — `ExternalRunSpec` / `ExternalResult` / `run()` + score parsing.
- `_compat.py` — stub drifted optional API SDKs so `bfcl_eval` imports under `--no-deps`.
- `__main__.py` — CLI.

> Multi-turn note: BFCL's new handler API drops the per-call `turn_type`; the handler
> infers multi-turn from tool-feedback in the message history. Single-turn (Non-Live /
> Live, which dominate Avg Acc) is exact; multi-turn uses the same ToolRL template.
