# LLMEval-Med

Native implementation of the released 667-item LLMEval-Med dataset.

- Candidate generation: `n=1`, sampling enabled with the upstream default temperature (`1.0`), max new tokens `2048`.
- Multi-turn groups are generated round-by-round; every later prompt includes the model's earlier answers exactly as chat history.
- Judge: `gpt-4o`; the upstream call does not set temperature or max tokens, so API defaults are preserved.
- Three judge repetitions are averaged per question before the `>=4` usability threshold.
- Primary metric: sample-weighted Overall Performance (`OP`, percent).

The paper's MTG score requires human ratings across five dimensions and a safety
veto. Because the released automatic evaluator cannot reproduce that human-only
number, the native automated result clearly reports the released GPT-4o
approximation and emits a warning.
