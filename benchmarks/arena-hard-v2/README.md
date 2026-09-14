# Arena-Hard-v2.0

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto).
Checked [config/arena-hard-v2.0.yaml](https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/config/arena-hard-v2.0.yaml)
and [config/api_config.yaml](https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/config/api_config.yaml).

The judge profile is `gpt-4.1, temperature=0, max_new_tokens=16000`.
Candidate generation is model/endpoint-specific in the official repository,
not one shared decoding tuple. Our `n=1, temperature=0, top_p=1,
max_new_tokens=8192` is a local candidate profile; the cap is not guaranteed
to be non-binding. This task evaluates the hard-prompt slice with its baseline
and style-controlled aggregation, not the separately configured creative slice.
Changing candidate budget or judge model changes the comparison protocol.

Native implementation of the official default 500-item `hard_prompt` slice.

- Candidate generation: `n=1`, temperature `0`, local output ceiling `8192`; upstream settings are model/endpoint-specific.
- Baseline: `o3-mini-2025-01-31`.
- Judge: `gpt-4.1`, temperature `0`, max tokens `16000`.
- Every answer is judged twice with A/B positions swapped; significant verdicts receive weight 3.
- Primary metric: official markdown+length style-controlled Bradley–Terry win rate, fitted together with the published GPT-4.1 judgment cohort snapshot. Raw win rate is also reported.

The separate 250-item creative-writing slice uses a different baseline and an
ensemble judge. It is intentionally not mixed into the hard-prompt score.
