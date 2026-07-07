# API-Bank (external benchmark)

Reproduces the **GD2PO API-Bank** numbers (per-level Acc / Format / Length for
levels 1/2/3) for ToolRL/GDPO-style models — i.e. models that emit
`<think>…</think>\n<tool_call>…</tool_call>\n<response>…</response>`.

## Why this is an *external* benchmark

The GD2PO pipeline fixes its own deterministic generation setup (greedy:
`temperature=0, top_p=1, seed=42, max_tokens=4096, max_model_len=4096`) and its own
output layout (`result.json` / `score_reward.json` / `leaderboard.json`), so — like
`benchmarks/bfcl/` — it is wrapped as an external benchmark with the same
`ExternalRunSpec` / `ExternalResult` / `run` shape instead of the native
`task.py`/`metrics.py` contract. Generation still runs through AetherEval's backend
abstraction (sglang default, vllm as in the reference); scoring is a faithful port of
the reference `evaluate_reward.py` + `aggregate_leaderboard.py`.

## Data

`data/level-{1,2,3}-api_processed.json` (399 / 67 / 131 samples), copied verbatim from
the GD2PO repo (`tool-calling/API_Bank/`). Each sample is
`{system, user, answer, other}`; the prompt is `[{system}, {user}]` rendered with the
model's chat template.

## Run

```bash
# base HF model:
python -m benchmarks.apibank \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/apibank-base

# a GDPO/GRPO checkpoint (any name + a local dir):
python -m benchmarks.apibank \
  --model rlla-gdpo --model-path /path/to/hf_ckpt \
  --output-dir outputs/apibank-gdpo --backend sglang --tp-size 1

# re-score existing generations only:
python -m benchmarks.apibank --model rlla-gdpo \
  --output-dir outputs/apibank-gdpo --skip-generation
```

`--levels` selects a subset (`1,2,3` default). `--backend vllm` matches the reference
engine exactly; sglang is the default (container ships sglang).

### Python API

```python
from pathlib import Path
from benchmarks.apibank.external import ExternalRunSpec, run

result = run(ExternalRunSpec(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir=Path("outputs/apibank-base"),
))
print(result.metrics, result.primary_score)   # primary_metric = "overall_acc"
```

## Output

```text
outputs/<run>/
  result.json         # per-sample generations keyed Level{1,2,3}_{idx} (reference schema)
  error.json          # samples whose generation failed
  score_reward.json   # result.json + per-sample score/format_score/length_score fields
  leaderboard.json    # per-level + overall record (reference aggregate_leaderboard.py schema)
  summary.json        # AetherEval-style: {metrics, primary_metric, primary_score}
```

## Metrics (faithful to `evaluate_reward.py`)

- `lv{1,2,3}_acc`, `overall_acc` — exact tool-name + parameter-dict match against the
  gold answer, over all samples of the level.
- `format_lv{1,2,3}_acc`, `overall_format_acc` — `<think>`/`<tool_call>`/`<response>`
  tag-structure check.
- `length_avg_lv{1,2,3}`, `overall_length_avg` — mean of
  `min(round(think_word_count / 512, 2), 1.0)`.
- plus raw counts (`correct_*`, `total_*`, `format_*`), `think_word_count_avg_*`, and
  `reward_avg_*` (= length avg), matching the reference leaderboard record.

The reference's `LENGTH_*`/`SCHEDULELENGTH` env overrides are never set by its eval
pipeline; their eval-time defaults (max 1.0, min 0.0, cap 512) are hard-coded here.

## Files

- `scoring.py` — faithful port of `evaluate_reward.py` scoring + leaderboard aggregation.
- `external.py` — `ExternalRunSpec` / `ExternalResult` / `run()` + deterministic generation.
- `__main__.py` — CLI.
- `data/` — the three level JSONs.

Scoring parity is covered by `tests/test_apibank_scoring.py` (no GPU needed):

```bash
python -m unittest tests.test_apibank_scoring
```
