# HumanEval+ Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [evalplus/evalplus](https://github.com/evalplus/evalplus).
Checked [codegen.py](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/codegen.py),
[DecoderBase](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/base.py), and
[vLLM provider](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/vllm.py).

Defaults retain greedy decoding with a local extended output budget:
`n=1, temperature=0, top_p=1, max_new_tokens=32768`.
The 32768-token ceiling includes reasoning and code; it is our evaluation choice,
not EvalPlus's reference decoder default of 768. The serving context must also
accommodate the prompt. This ceiling does not require every response to use it.

The chat prompt asks for reasoning plus fenced code, which is not EvalPlus's
prompt. Local code-block assembly is followed by EvalPlus `sanitize`; execution
uses EvalPlus commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, with that
revision's native tolerances, special oracles and time limits.

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
  - assemble imports, helpers and definitions across code blocks, retaining the
    last function/class definition and dropping top-level usage examples, then use
    EvalPlus `sanitize(..., entrypoint=...)` to retain the implementation dependencies
  - indented function bodies are parsed as `prompt + body`; the final body can
    replace an earlier complete draft, while prompt imports/helpers remain available
  - compute reference outputs from the canonical solution (cached per process)
  - run `base_input` and `plus_input` using EvalPlus's subprocess checker
- The pinned upstream commit includes the `find_zero` fix (#241); no local oracle
  patch is applied. It uses a 4-second minimum per-test time limit, up from
  0.3.1's 1 second. Serial and spawned workers use the same installed evaluator.
- HumanEval/32's canonical Newton solver is numerically sensitive to the Python
  runtime. In our audit, Python 3.10.14 passed all 788 Plus inputs; Python 3.12.4
  failed 7. Python 3.12's changed float `sum` changes the iteration trajectory;
  this is not evidence that the tolerance should be relaxed. Python 3.12 remains
  supported: keep its native arithmetic and the official inputs/tolerance, and
  use the same runtime for all compared models. Generated solutions are still
  checked by the official root-residual oracle, not by matching canonical outputs.
- `score` / `is_pass` requires both suites to pass. If base fails, Plus is skipped.
- `accuracy_base` reports base tests; `accuracy_plus` requires base and Plus.
- Records identify the checker with
  `meta.scoring_protocol=evalplus-26d6d00`.

Older scores ran only `test + check(entry_point)` and copied that result into
both base and Plus fields. They are not verified HumanEval+ scores. Re-score
saved generations with `--eval-only` (same model-name, output-dir and run-id);
no generation or training is needed. Existing result files are not automatically
migrated. Follow the repository [installation instructions](../../README.md#install)
first; in AetherRL Docker use `python -m pip install --no-deps -e .`.

Generated programs are untrusted: run evaluation in an isolated environment
without credentials or valuable files; EvalPlus's guard is not a security sandbox.

Reported metrics include:
- `accuracy` (alias of `accuracy_plus`)
- `accuracy_plus`, `accuracy_base`
- `accuracy_stderr`, `accuracy_plus_stderr`, `accuracy_base_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
