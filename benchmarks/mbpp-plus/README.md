# MBPP+

Pinned dataset: **MBPP+ v0.2.0, 378 tasks**, full release (not NoExtreme).
Pinned evaluator: **EvalPlus `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`**, installed separately with `--no-deps`
(already included in AetherRL Docker; see [installation](../../README.md#install)).

## Offline use

The bundled `data/eval.jsonl` contains the prompt, canonical solution, base and
Plus inputs, and tolerance. To download it again on a connected machine:

```bash
python -m benchmarks.mbpp-plus.prepare_data
```

With dependencies and the model also available locally, both generation and
scoring run without network access or an external judge:

```bash
aethereval --model /path/to/policy --tasks mbpp-plus --output-dir outputs
```

## Official source and protocol

- Prompt is the AetherRL code-training prompt shared with HumanEval+ and
  LiveCodeBench (`benchmark_utils/code_prompt.py`): one user turn, no system message,
  "Please think step by step, then write the complete solution.", the function
  interface line naming the tested callable, and one fenced Python block. Base/Plus
  execution and aggregation are shared with HumanEval+ in `benchmark_utils/evalplus.py`. The question
  retains MBPP+'s original specification and examples; no reference solution or
  private test inputs are included. There is no assistant/code prefill.
  The local backend applies the model's chat template. This is a local reasoning
  chat profile, not the unmodified EvalPlus chat or completion prompt.
- Greedy defaults: `n=1`, `temperature=0`, `top_p=0.95`, `max_new_tokens=32768`.
  `top_p` follows the official chat request helper (temperature zero is greedy).
  The output budget is a local extension of the reference decoder's 768-token
  default. It accommodates reasoning plus code without forcing a model-specific
  thinking mode. The serving context must also
  fit the prompt; EOS can end generation before the ceiling.
- Use the official `sanitize`, `mbpp_deserialize_inputs`, and
  `untrusted_check(dataset="mbpp")`, including special oracles and official time limits.
  `sanitize` runs with an output-identical, bounded `code_extract`
  (`benchmark_utils/evalplus_sanitize.py`) that avoids upstream's cubic scan, which
  can grind for hours on a degenerate truncated response; a deterministic parse
  budget backstops adversarial inputs, and a differential test checks the result
  against upstream.
  Reference execution also honors `MBPP_OUTPUT_NOT_NONE_TASKS`.
  This revision uses a 4-second minimum per-test time limit (0.3.1 used 1 second);
  re-score saved generations with `--eval-only` after upgrading.
- Known upstream defect: `sanitize` drops the recursive generator helper `adjac`
  from the canonical Mbpp/630 solution, so that reference fails its base tests.
- Primary `pass@1` and `accuracy_plus` require both base and Plus tests to pass.
  `accuracy_base` is reported separately on the same 378 tasks, not the original
  full MBPP benchmark. A base failure skips Plus execution without changing pass/fail.
- No runtime dataset downloads: scoring reads raw local inputs and computes
  reference outputs locally, cached per process.
- EvalPlus limits each checker to 4 GiB of address space. Scoring always runs in
  spawned workers with BLAS threads capped at 1 (unless already set in the
  environment), including at `--num-proc 1` (one worker), so that budget does not
  depend on core count or `--num-proc`. Each check then starts a fresh
  interpreter: on a 64-core node the 378 canonical solutions take about 220 s at
  `--num-proc 1` and 38 s at `--num-proc 16`, so use a larger `--num-proc`.
  Earlier runs failed the canonical Mbpp/255 on a 64-core node; re-score earlier
  checkpoints with `--eval-only` before comparing against them.

Execute generated programs in an isolated environment without credentials or
valuable files. EvalPlus's reliability guard is not a security sandbox.
