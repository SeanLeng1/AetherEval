# TriviaQA

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [mandarjoshi90/triviaqa](https://github.com/mandarjoshi90/triviaqa),
including its [evaluation documentation](https://github.com/mandarjoshi90/triviaqa/blob/master/README.md).

The original release is a reading-comprehension benchmark and does not impose
one temperature/top-p/output limit for present-day generative chat models.
Our `unfiltered.nocontext` validation task is a closed-book adaptation with
`n=1, temperature=0, top_p=1, max_new_tokens=64`.
The short-answer cap is local, not an official benchmark requirement.
Do not compare these results to context-provided or verified-subset scores
without matching the data, evidence access and answer normalization.

This task uses the public `unfiltered.nocontext` validation split and reports normalized
exact match over the complete submitted answer string, with token F1 as a secondary
diagnostic. The public validation split is used because labeled test answers are not
part of the normal reproducible evaluator.

```bash
python benchmarks/triviaqa/prepare_data.py
```

If the shared Hugging Face cache is full, pass a writable temporary directory such as
`--cache-dir /tmp/aethereval-hf-cache`.
