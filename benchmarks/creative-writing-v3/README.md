# Creative Writing Bench V3

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [EQ-bench/creative-writing-bench](https://github.com/EQ-bench/creative-writing-bench).
Checked [README](https://github.com/EQ-bench/creative-writing-bench/blob/c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b/README.md)
and [core/conversation.py](https://github.com/EQ-bench/creative-writing-bench/blob/c7c3ceef54c40a8ae02dc1c2e1a5e40970fe5c0b/core/conversation.py).

The rubric-only profile uses `temperature=0.7, min_p=0.1,
max_new_tokens=12000`; 32 prompts are expanded across three seed modifiers.
AetherEval explicitly leaves top-p/top-k unfiltered and uses `n=1` per
expanded item. Judge settings are `claude-sonnet-4-6, temperature=0,
max_new_tokens=4096`, as recommended for leaderboard judging.
The no-Elo rubric score is not the pairwise Elo leaderboard result.
Upstream provider-specific reasoning overrides do not become universal
defaults for every locally served model.

Native rubric-score implementation of Creative Writing Bench V3.

- 32 prompts × the first 3 seed modifiers = 96 generated pieces.
- Candidate generation: temperature `0.7`, min-p `0.1`, max new tokens `12000`, `n=1` per expanded row.
- Responses shorter than 500 characters are retried up to three total attempts.
- Judge: `claude-sonnet-4-6`, temperature `0`, max tokens `4096`.
- Primary metric: `eqbench_creative_score` (rubric mean scaled from 0–20 to 0–100), with the nine negative criteria inverted.

This corresponds to upstream `--iterations 3 --no-elo`. The optional pairwise Elo
stage requires a separate bank of other systems' outputs and is intentionally not
mixed into the native absolute rubric metric.
