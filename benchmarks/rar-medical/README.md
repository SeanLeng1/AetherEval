# RaR Medical

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official release: [ScaleAI/RaR-Medicine](https://huggingface.co/datasets/ScaleAI/RaR-Medicine).
Official methodology: [Rubrics as Rewards](https://arxiv.org/html/2507.17746v1).
No standalone official inference repository for this held-out rubric task was
verified; the data release and paper are the authoritative sources.

Defaults `n=1, temperature=0, top_p=0.7, max_new_tokens=8192` and the
Gemma-4 judge (`temperature=1, top_p=1, max_new_tokens=4096`, thinking off)
are a **local RaR/CriPO-style evaluation profile**.
At temperature zero, top-p does not make decoding stochastic.
The RaR paper evaluates medicine on HealthBench-1k, not this held-out
RaR-Medicine rubric score. Its training judge is GPT-4o-mini.
This task's Gemma judge, split and rubric aggregation must not be presented as
that paper's official benchmark result.

Native evaluation of the official `ScaleAI/RaR-Medicine` test split.

Prepare the independent AetherEval JSONL file:

```bash
python benchmarks/rar-medical/prepare_data.py
```

The processor downloads the release itself; it does not read AetherRL data. It
validates and deterministically deduplicates the official test split, then
writes `data/eval.jsonl` using AetherEval's regular task schema. Training-data
decontamination is deliberately not allowed to delete evaluation prompts.

The public release contains 2,242 raw test rows. CriPO's reported 1,936-row
evaluation set is a different, model-dependent subset that removes examples on
which Qwen3-4B already scores above 0.9; this processor does not silently apply
that difficulty filter.

Each candidate response is judged once against all of its criteria with the
RaR/CriPO `PRESENT`/`NOT_PRESENT` prompt. The primary `score` is the normalized
positive-weight sum with `Essential=1.0`, `Pitfall=0.9`, `Important=0.7`, and
`Optional=0.3`. A grader failure produces a zero score and is reported by
`judge_failure_rate` rather than aborting evaluation.
