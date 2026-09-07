# Arena-Hard-v2.0

Native implementation of the official default 500-item `hard_prompt` slice.

- Candidate generation: `n=1`, temperature `0`; upstream makes max tokens model/endpoint-specific, so Aether uses a non-binding local ceiling of `8192`.
- Baseline: `o3-mini-2025-01-31`.
- Judge: `gpt-4.1`, temperature `0`, max tokens `16000`.
- Every answer is judged twice with A/B positions swapped; significant verdicts receive weight 3.
- Primary metric: official markdown+length style-controlled Bradley–Terry win rate, fitted together with the published GPT-4.1 judgment cohort snapshot. Raw win rate is also reported.

The separate 250-item creative-writing slice uses a different baseline and an
ensemble judge. It is intentionally not mixed into the hard-prompt score.
