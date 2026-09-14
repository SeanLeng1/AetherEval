# ResearchQA

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [realliyifei/ResearchQA](https://github.com/realliyifei/ResearchQA).
Checked [compute_coverage.py](https://github.com/realliyifei/ResearchQA/blob/747a9a1330f097a0e20672240cb47e3cf02500ae/compute_coverage.py)
and the [paper](https://arxiv.org/abs/2509.00496).
This is the survey-mined ResearchQA release, not similarly named benchmarks.

The released coverage judge uses `gpt-4.1-mini, temperature=0`,
rubric batches of eight, and no explicit API max-token limit.
AetherEval sets top-p 1 and a local 4096-token judge cap.
Candidate defaults `n=1, temperature=0, top_p=1, max_new_tokens=2048`
are a local direct-answer profile; the paper's approximate 250-word instruction
does not establish a tokenizer-independent 2048-token limit.
A direct-answer run without retrieval must not be represented as an equivalent
deep-research agent evaluation.

Native implementation of rubric coverage on the 3,750-item ResearchQA test set.

- Candidate task protocol: citation-supported answer, source cutoff at the item's `date`, approximately 250 words, `n=1`, temperature `0`.
- `max_new_tokens=2048` is a local output ceiling; the paper specifies the 250-word instruction but no common API max-token value.
- Judge: `gpt-4.1-mini`, temperature `0`, rubric batches of 8, three format attempts.
- Primary metric: normalized rubric `coverage` on a 0–100 scale.

The paper states the constraints but does not release the exact candidate prompt
string. This implementation expresses those constraints directly and records the
rendered prompt in every prediction for auditability.
