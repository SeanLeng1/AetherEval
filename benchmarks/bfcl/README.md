# BFCL V3 (external benchmark)

Runs the official Berkeley Function Calling Leaderboard V3 generation loop and
evaluator for ToolRL/GDPO-style models that emit
`<think>…</think>\n<tool_call>…</tool_call>\n<response>…</response>`.

## Why it is external

BFCL owns a multi-turn tool-execution loop and specialized AST/execution checkers, so
it does not fit AetherEval's single-shot `task.py` + `metrics.py` contract. The CLI
still exposes it normally as `--tasks bfcl`; `benchmarks/bfcl/external.py` drives the
official package in process and writes the standard AetherEval output files.

The required release is:

```bash
pip install --no-deps "bfcl-eval==2025.6.8"
```

The runtime must also contain the real Tree-sitter Python, Java, and JavaScript
parsers. `_compat.py` may stub unused online-provider SDKs, but it never stubs scoring
dependencies.

## Scoring equivalence

AetherEval makes one narrow correction for ToolRL-style JSON serialization. If the
official tool schema declares an `integer`, `float`, or `boolean`, the corresponding
canonical JSON scalar encoded as a string is normalized before official scoring or
multi-turn execution. Examples include `"2"` → `2`, `"2.5"` → `2.5`, and `"true"` →
`true`.

The rule is recursive for schema-declared array/tuple items and dict properties. It
does not coerce string/any parameters, entire containers encoded as strings,
undeclared fields, malformed values such as `"True"`, missing arguments, or function
names. This prevents representation-only false negatives without making the checker
generally lenient.

## Defaults and execution

The default collections are `live,non_live,multi_turn`, which together are the full
BFCL V3 benchmark. Generation settings live under `bfcl` in
`configs/task_defaults.yaml`:

- `n: 1`: one generation at each BFCL interaction.
- `num_repeats: 4`: four complete benchmark runs, averaged at the end.
- `temperature: 0.001`, `top_p: 1.0`, `top_k: -1`.
- `max_new_tokens: 4096`.

`n` and `num_repeats` are intentionally distinct. Seeds are `0,1,2,3` by default;
`--seed S` changes them to `S,S+1,S+2,S+3`. Use `--num-repeats 1` for a diagnostic
run.

```bash
aethereval \
  --tasks bfcl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir outputs \
  --dp-size 8 \
  --tp-size 1
```

`--model` is always the actual Hugging Face ID or checkpoint path. Optional
`--model-name` only controls the AetherEval output label, including when the model ID
contains `/` or `_`.

`--categories` accepts the official V3 collections `all`, `live`, `non_live`,
`multi_turn`, `single_turn`, `ast`, `python`, and `non_python`, or individual V3
categories such as `simple`, `live_parallel`, and `multi_turn_miss_param`. Omitting it
selects `live,non_live,multi_turn`.

With SGLang, `--dp-size` is the replica count and `--tp-size` is tensor parallelism
per replica. AetherEval starts its managed SGLang Model Gateway and reuses it across
all repeats. BFCL requests are concurrent; `--num-threads` defaults to
`max(16, 16 * dp_size)`, capped at 100.

Deterministic context-length errors are not retried and score zero. Connection errors,
429s, and transient 5xx responses retain bounded retries. Use `--bfcl-verbose` to show
BFCL's normally filtered per-turn logs.

Generation resumes existing JSONL results unless `--overwrite` is set. If a process
was interrupted during its final JSONL write, AetherEval saves the incomplete bytes as
`*.corrupt-tail`, truncates only that final record, and resumes it. Earlier corruption
is reported and never silently repaired.

`--bfcl-context-length` and repeatable `--bfcl-sglang-arg KEY=VALUE` affect only BFCL.
No RoPE scaling is enabled automatically: extending a checkpoint beyond its naturally
supported context can change accuracy.

## Output and metrics

```text
outputs/<model_name>/<run_id>/bfcl/
  predictions.jsonl
  result/run_01/ ... result/run_04/
  score/run_01/  ... score/run_04/
  summary.json
```

`summary.json` contains the mean across complete repeats. The canonical metric set is:

| metric | definition |
|---|---|
| `live_acc` | official V3 `Live Acc` |
| `live_format` | ToolRL format rate for Live outputs |
| `non_live_acc` | official V3 `Non-Live AST Acc` |
| `non_live_format` | ToolRL format rate for Non-Live outputs |
| `multi_turn_acc` | official V3 `Multi Turn Acc` |
| `multi_turn_format` | ToolRL format rate for Multi-Turn outputs |
| `overall_acc` | official V3 `Overall Acc` |
| `overall_format` | unweighted mean of the three section format rates |

Official V3 defines `Overall Acc` as the unweighted mean of Live Overall, Non-Live
Overall, and Multi-Turn Overall accuracy. The exposed `non_live_acc` is the paper-style
AST summary, whereas Non-Live Overall also includes irrelevance detection; consequently,
the three displayed split columns do not in general average exactly to `overall_acc`.
Format is an AetherEval comparison metric, not an official BFCL CSV column;
`overall_format` is the unweighted mean of the three displayed section format rates.
Ordinary calling categories require `<tool_call>`, irrelevance categories require
`<response>`, and multi-turn expectations come from each turn's official ground truth.
Tool-execution steps require `<tool_call>` and a completed turn ends with `<response>`.

`predictions.jsonl` follows AetherEval's normal row shape. `gen_idx` identifies the
repeat and `meta.evaluation_repeat` is its one-based index; the original BFCL record is
preserved under `meta.bfcl_record`.

## Python API

```python
from pathlib import Path

from benchmarks.bfcl.external import ExternalRunSpec, run

result = run(
    ExternalRunSpec(
        model="Qwen/Qwen2.5-3B-Instruct",
        output_dir=Path("outputs/bfcl"),
        backend="sglang",
        dp_size=8,
        tp_size=1,
    )
)
print(result.metrics, result.primary_score)  # primary_metric = "overall_acc"
```
