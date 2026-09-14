# HumanEval+ Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [evalplus/evalplus](https://github.com/evalplus/evalplus).
Checked [codegen.py](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/codegen.py),
[DecoderBase](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/base.py), and
[vLLM provider](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/vllm.py).

Defaults now follow the documented greedy profile:
`n=1, temperature=0, top_p=1, max_new_tokens=768`.
768 is the reference decoder's default output budget, not a claim that every
reasoning model finishes within it. Override the budget explicitly for extended
reasoning experiments. The former `n=10, temperature=0.6, top_p=0.95, 4096`
profile was a local sampled evaluation, not this reference profile.

The chat prompt asks for reasoning plus fenced code and uses local extraction.
These are not identical to EvalPlus prompting and sanitization. Test execution
now uses EvalPlus 0.3.1, including its tolerances, special oracles and time limits.

## Files

```text
benchmarks/humaneval-plus/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  prepare_data.py
```

## Data

- Source: EvalPlus HumanEval+ release (`v0.1.10`)
- Local offline file: `data/eval.jsonl`
- Regeneration: `prepare_data.py`

Each row keeps EvalPlus fields (`task_id`, `prompt`, `entry_point`, `canonical_solution`, `base_input`, `plus_input`, `atol`, ...).

## Prompting

- Implemented in `task.py`
- Uses chat-style prompt with explicit sections:
  - system instruction for Python code completion
  - user sections: `### Question`, `### Format`, `### Answer`
- `### Format` asks for short reasoning plus completed function inside a fenced Python block.
- The framework applies the model chat template.

## Metrics

- Implemented in `metrics.py`
- For each generation:
  - extract the final code candidate (prefer answer block / fenced code)
  - execute `prompt + continuation`
  - compute reference outputs from the canonical solution (cached per process)
  - run `base_input` and `plus_input` using EvalPlus's subprocess checker
- `score` / `is_pass` requires both suites to pass. If base fails, Plus is skipped.
- `accuracy_base` reports base tests; `accuracy_plus` requires base and Plus.
- Records identify the checker with `meta.scoring_protocol=evalplus-0.3.1`.

Older scores ran only `test + check(entry_point)` and copied that result into
both base and Plus fields. They are not verified HumanEval+ scores. Re-score
saved generations with `--eval-only` (same model-name, output-dir and run-id);
no generation or training is needed. Existing result files are not automatically
migrated. Install the updated dependencies first (`pip install -e .`).

Generated programs are untrusted: run evaluation in an isolated environment
without credentials or valuable files; EvalPlus's guard is not a security sandbox.

Reported metrics include:
- `accuracy` (alias of `accuracy_plus`)
- `accuracy_plus`, `accuracy_base`
- `accuracy_stderr`, `accuracy_plus_stderr`, `accuracy_base_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
