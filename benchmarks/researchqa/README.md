# ResearchQA

Native implementation of rubric coverage on the 3,750-item ResearchQA test set.

- Candidate task protocol: citation-supported answer, source cutoff at the item's `date`, approximately 250 words, `n=1`, temperature `0`.
- `max_new_tokens=2048` is a non-binding safety ceiling; the paper specifies the 250-word instruction but no common API max-token value.
- Judge: `gpt-4.1-mini`, temperature `0`, rubric batches of 8, three format attempts.
- Primary metric: normalized rubric `coverage` on a 0–100 scale.

The paper states the constraints but does not release the exact candidate prompt
string. This implementation expresses those constraints directly and records the
rendered prompt in every prediction for auditability.
