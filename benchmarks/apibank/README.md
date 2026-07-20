# API-Bank

Native AetherEval task for the GD2PO API-Bank setup. It reports the per-level
correctness, format, and diagnostic length-reward metrics for ToolRL/GDPO-style models that emit
`<think>...</think>`, `<tool_call>...</tool_call>`, and/or `<response>...</response>`.

## Data

`data/eval.jsonl` contains 597 samples from GD2PO's `tool-calling/API_Bank/` data:
399 level-1, 67 level-2, and 131 level-3 samples. The original
`level-{1,2,3}-api_processed.json` files are kept as source snapshots, and
`prepare_data.py` regenerates the native JSONL file from them.

Each sample stores `{system, user, answer, other}`. The prompt is the two-message chat
conversation `[{role: system}, {role: user}]`, rendered by the selected AetherEval
backend with the model's chat template.

## Run

```bash
aethereval \
  --backend sglang \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tasks apibank \
  --output-dir outputs \
  --tp-size 1
```

APIBank defaults live in `configs/task_defaults.yaml`:

```yaml
apibank:
  n: 1
  max_new_tokens: 4096
  temperature: 0.0
  top_p: 1.0
```

That matches the reference greedy generation setup in the parts that belong to the
generation request. Backend context length remains a normal AetherEval runtime setting
such as `--max-model-len` for vLLM or `--context-length` for SGLang.

## Output

The task uses the standard native output layout:

```text
outputs/<run_id>/apibank/
  predictions.jsonl
  run_config.json
  summary.json
```

Per-generation `meta` in `predictions.jsonl` includes `correct_score`,
`format_score`, `length_score`, and `think_word_count`.

## Metrics

- `lv{1,2,3}_acc`, `overall_acc` — exact tool-name + parameter-dict match against the
  gold answer, using the strict GD2PO tag parser (single `<tool_call>` block).
- `loose_{lv1,lv2,lv3}_acc`, `loose_overall_acc` — same match, but tool calls are
  parsed with **ToolRL's own `generate.py` parser** (last `<tool_call>` block, per-line
  JSON, no closing-tag guard). This is the ToolRL-aligned number on the same
  generations; it recovers calls in an unclosed last block but still misses a correct
  call placed in a non-last block (exactly as ToolRL scores it). The residual to
  ToolRL's paper number is the sglang-vs-vLLM engine gap, not scoring. Report alias:
  `LooseCorrectAcc.`, `Loose Level {1,2,3} Acc.`.
- `format_lv{1,2,3}_acc`, `overall_format_acc` — tag-structure check.
- `length_avg_lv{1,2,3}`, `overall_length_avg` — mean of
  `min(round(think_word_count / 512, 2), 1.0)`, where `think_word_count` is the
  whitespace-split count of the `<think>...</think>` content, matching GD2PO's
  APIBank eval/reward code.
- Report aliases: `Level 1 Acc.`, `Level 2 Acc.`, `Level 3 Acc.`, `CorrectAcc.`,
  `FormatAcc.`, `LengthRew.`, `LengthReward`, and `Overall`.
- `Overall = CorrectAcc. / 100 + FormatAcc. / 100` and is the AetherEval
  `PRIMARY_METRIC` (range `0` to `2`). `LengthRew.` remains available as a
  diagnostic metric but does not contribute to `Overall`.

Raw counts (`correct_*`, `total_*`, `format_*`), `think_word_count_avg_*`, and
`reward_avg_*` are also reported for parity with the GD2PO leaderboard layer.

The reference's `LENGTH_*`/`SCHEDULELENGTH` env overrides are never set by its eval
pipeline; their eval-time defaults (max 1.0, min 0.0, cap 512) are hard-coded here.

## Files

- `task.py` — native sample loading and prompt construction.
- `metrics.py` — native scoring/aggregation entry points.
- `scoring.py` — faithful port of GD2PO's scoring and leaderboard aggregation logic.
- `prepare_data.py` — source JSON to native JSONL conversion.
- `data/` — native JSONL plus the original level JSON snapshots.

Scoring parity is covered by `tests/test_apibank_scoring.py` (no GPU needed):

```bash
PYTHONPATH=$PWD python tests/test_apibank_scoring.py
```
