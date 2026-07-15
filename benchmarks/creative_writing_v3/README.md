# Creative Writing Bench V3

Native rubric-score implementation of Creative Writing Bench V3.

- 32 prompts × the first 3 seed modifiers = 96 generated pieces.
- Candidate generation: temperature `0.7`, min-p `0.1`, max new tokens `12000`, `n=1` per expanded row.
- Responses shorter than 500 characters are retried up to three total attempts.
- Judge: `claude-sonnet-4-6`, temperature `0`, max tokens `4096`.
- Primary metric: `eqbench_creative_score` (rubric mean scaled from 0–20 to 0–100), with the nine negative criteria inverted.

This corresponds to upstream `--iterations 3 --no-elo`. The optional pairwise Elo
stage requires a separate bank of other systems' outputs and is intentionally not
mixed into the native absolute rubric metric.
