# RaR Science

Native evaluation of the official `ScaleAI/RaR-Science` test split.

Prepare the independent AetherEval JSONL file:

```bash
python benchmarks/rar-science/prepare_data.py
```

The processor downloads the release itself; it does not read AetherRL data. It
validates and deterministically deduplicates the official test split, then
writes `data/eval.jsonl` using AetherEval's regular task schema. Training-data
decontamination is deliberately not allowed to delete evaluation prompts.

CriPO's reported 1,365-row evaluation set is a model-dependent subset that
removes examples on which Qwen3-4B already scores above 0.9; this processor does
not silently apply that difficulty filter to the public test split.

Each candidate response is judged once against all of its criteria with the
RaR/CriPO `PRESENT`/`NOT_PRESENT` prompt. The primary `score` is the normalized
positive-weight sum with `Essential=1.0`, `Pitfall=0.9`, `Important=0.7`, and
`Optional=0.3`. A grader failure produces a zero score and is reported by
`judge_failure_rate` rather than aborting evaluation.
