# BFCL V3 (external benchmark)

Runs the official Berkeley Function Calling Leaderboard V3 generation loop and
evaluator. A selectable handler profile separates the model's output protocol from
its architecture and tokenizer chat template.

- `toolrl` (default) evaluates ToolRL/GDPO-style models that emit
  `<think>…</think>\n<tool_call>…</tool_call>\n<response>…</response>`.
- `official` reuses the prompt and decoder of an exact prompt-mode model registration
  from the installed BFCL V3 package.

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

## Standard scoring

Both handler profiles use BFCL V3's official checker without value normalization.
For example, a schema-declared integer emitted as the JSON string `"2"` is not changed
to the JSON number `2` before scoring or multi-turn execution. AetherRL separately
handles stringified values in its training reward because some ToolRL ground-truth
annotations are stringified; that training-data correction is intentionally not part
of the benchmark protocol.

## Defaults and execution

The default collections are `live,non_live,multi_turn`, which together are the full
BFCL V3 benchmark. Generation settings live under `bfcl` in
`configs/task_defaults.yaml`:

- `n: 1`: one generation at each BFCL interaction.
- `num_repeats: 4`: four complete benchmark runs, averaged at the end.
- `handler: toolrl`: ToolRL prompt/output protocol.
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

Use `--bfcl-handler official` for an unmodified model that has an exact prompt-mode
entry in `bfcl-eval==2025.6.8`, for example:

```bash
aethereval \
  --tasks bfcl \
  --model google/gemma-3-12b-it \
  --bfcl-handler official \
  --output-dir outputs
```

This pinned BFCL release includes official Gemma 3 handlers, but not Gemma 4 or
Qwen2.5 model registrations. ToolRL-trained models and arbitrary local checkpoints
should use `toolrl`. Native function-calling (`is_fc_model=True`) registrations are
also rejected explicitly because they require a different structured transport.

The ToolRL instruction and serialized dialogue history are model-independent content.
Their outer chat envelope is rendered with the selected model tokenizer's
`apply_chat_template(..., add_generation_prompt=True)`; Qwen therefore receives
ChatML, while Gemma receives Gemma turn markers. If a template rejects the `system`
role (as Gemma-family templates commonly do), AetherEval folds the complete system
instruction into the user message before applying that same model template. A model
without a usable chat template fails clearly instead of silently receiving Qwen
tokens. `--enable-thinking` and `--no-enable-thinking` are forwarded to compatible
BFCL tokenizer templates too.

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

`summary.json` contains the mean across complete repeats. In `toolrl` mode, the
canonical metric set is:

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
The `official` profile reports only BFCL's official accuracy metrics because ToolRL
tag-format rates do not apply to an official model-specific output protocol.

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
