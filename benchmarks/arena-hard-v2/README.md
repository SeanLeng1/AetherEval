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
to be non-binding. Both upstream slices are evaluated, each with its own baseline,
judge system prompt and aggregation, and they are never mixed into one score.
Changing candidate budget or judge model changes the comparison protocol.

Native implementation of the 500-item `hard_prompt` slice and the 250-item
`creative_writing` slice (750 prompts).

- Candidate generation: `n=1`, temperature `0`, local output ceiling `8192`; upstream settings are model/endpoint-specific.
- Baselines: `o3-mini-2025-01-31` (hard prompt), `gemini-2.0-flash-001` (creative writing).
- Judge: `gpt-4.1`, temperature `0`, max tokens `16000`.
- Every answer is judged twice with A/B positions swapped; significant verdicts receive weight 3.
- Primary metric: official markdown+length style-controlled Bradley–Terry win rate, fitted together with the published GPT-4.1 judgment cohort snapshot. Raw win rate is also reported.

- `creative_writing_win_rate`: upstream `show_result.py --category creative_writing`
  aggregation, i.e. no style control, mean of 100 bootstrap means with 5%/95%
  quantiles, judged with the creative-writing system prompt (no "generate your own
  answer" step). Upstream's best configuration ensembles GPT-4.1 and Gemini-2.5
  judgments for this slice; this task uses the single configured judge.
- Reasoning blocks (`<think>...</think>`) are removed before judging and before the
  style features are measured, as upstream does via `end_think_token`.
