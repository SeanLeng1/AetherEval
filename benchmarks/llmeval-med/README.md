# LLMEval-Med

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [llmeval/LLMEval-Med](https://github.com/llmeval/LLMEval-Med).
Checked [evaluate/Answer.py](https://github.com/llmeval/LLMEval-Med/blob/a5a90f9273994334073c539136ce85f0815fffbc/evaluate/Answer.py)
and [evaluate/Evaluate.py](https://github.com/llmeval/LLMEval-Med/blob/a5a90f9273994334073c539136ce85f0815fffbc/evaluate/Evaluate.py).

The candidate script sets `do_sample=True, max_new_tokens=2048` but inherits
temperature/top-p/top-k from the selected Transformers model. Therefore our
`n=1, temperature=1, top_p=1` is an explicit unfiltered local profile,
**not a guaranteed reproduction of every upstream model's generation config**.
The judge uses GPT-4o without explicit sampling or token limits. Our
`temperature=1, top_p=1, max_new_tokens=4096` makes those settings explicit
and adds a local cap. Three grading repetitions are retained; the automatic
usability metric is not the physician-rated paper metric.

Native implementation of the released 667-item LLMEval-Med dataset.

- Candidate generation: `n=1`, sampling enabled, explicit local temperature `1.0`, max new tokens `2048`.
- Multi-turn groups are generated round-by-round; every later prompt includes the model's earlier answers exactly as chat history.
- Judge: `gpt-4o`; AetherEval explicitly sets temperature `1.0`, top-p `1.0`, and a local `4096`-token cap where upstream omits these fields.
- Three judge repetitions are averaged per question before the `>=4` usability threshold. As in upstream `Aggregate.py`, only repetitions with a valid 0-5 score enter the average; a question with no valid repetition counts as not usable.
- Primary metric: sample-weighted Overall Performance (`OP`, percent).

The paper's MTG score requires human ratings across five dimensions and a safety
veto. Because the released automatic evaluator cannot reproduce that human-only
number, the native automated result clearly reports the released GPT-4o
approximation and emits a warning.
