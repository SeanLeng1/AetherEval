# HealthBench

Native implementation of the main OpenAI `simple-evals` HealthBench release.

- Data: `2025-05-07-06-14-12_oss_eval.jsonl`.
- Candidate generation: `n=1`, temperature `0.5`, max new tokens `2048`.
- Judge: `gpt-4.1-2025-04-14`, temperature `0.5`, max tokens `2048`, one independent call per rubric item.
- Primary metric: clipped mean rubric score in `[0, 1]`, including negative-point rubric behavior and tag-level metrics.

The implementation is native; online judge inference uses LiteLLM. Set the selected
provider's normal API-key environment variable (for the default, `OPENAI_API_KEY`).
`AETHEREVAL_JUDGE_BASE_URL` is only needed for an OpenAI-compatible gateway.
