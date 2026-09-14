# HealthBench

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [openai/simple-evals](https://github.com/openai/simple-evals).
Checked [simple_evals.py](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/simple_evals.py),
[chat_completion_sampler.py](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/sampler/chat_completion_sampler.py),
and [healthbench_eval.py](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/healthbench_eval.py).

Candidate defaults `n=1, temperature=0.5, top_p=1, max_new_tokens=2048`
match the non-reasoning GPT-4.1-style sampler configuration, not every model in
simple-evals. Judge defaults match its GPT-4.1 grading path:
`gpt-4.1-2025-04-14, temperature=0.5, top_p=1, max_new_tokens=2048`.
Upstream also has separate reasoning-model profiles and an optional newer
grader; neither is silently substituted here. Results are for the main release,
not HealthBench-Hard or a model-dependent subset.

Native implementation of the main OpenAI `simple-evals` HealthBench release.

- Data: `2025-05-07-06-14-12_oss_eval.jsonl`.
- Candidate generation: `n=1`, temperature `0.5`, max new tokens `2048`.
- Judge: `gpt-4.1-2025-04-14`, temperature `0.5`, max tokens `2048`, one independent call per rubric item.
- Primary metric: clipped mean rubric score in `[0, 1]`, including negative-point rubric behavior and tag-level metrics.

The implementation is native; online judge inference uses LiteLLM. Set the selected
provider's normal API-key environment variable (for the default, `OPENAI_API_KEY`).
`AETHEREVAL_JUDGE_BASE_URL` is only needed for an OpenAI-compatible gateway.
