# NQ-Open

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [google-research-datasets/natural-questions](https://github.com/google-research-datasets/natural-questions),
specifically the [nq_open release](https://github.com/google-research-datasets/natural-questions/tree/master/nq_open).

The release specifies open-domain question/answer data, not one modern chat
model's decoding configuration. Defaults `n=1, temperature=0, top_p=1,
max_new_tokens=64` are a local closed-book, short-answer profile.
They are not a protocol for the original NQ passage/long-answer task or a
retrieval-augmented system. Keep prompt wording, retrieval access and normalized
exact-match scoring fixed across comparisons. The 64-token cap can truncate
reasoning-model answers; any enlarged budget should be reported explicitly.

This task uses the 3,610-example public NQ-Open development set and normalized exact
match over the complete submitted answer string. The original test labels are not
public, so reports should call this split
**NQ-Open dev**, rather than NQ test. Token F1 is included as a secondary diagnostic.

```bash
python benchmarks/nq-open/prepare_data.py
```
