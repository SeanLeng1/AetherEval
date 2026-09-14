# MBPP+

Pinned dataset: **MBPP+ v0.2.0, 378 tasks**, full release (not NoExtreme).
Pinned evaluator: **EvalPlus 0.3.1**, installed separately with `--no-deps`
(already included in AetherRL Docker; see [installation](../../README.md#install)).

## Offline use

The bundled `data/eval.jsonl` contains the prompt, canonical solution, base and
Plus inputs, and tolerance. To download it again on a connected machine:

```bash
python benchmarks/mbpp-plus/prepare_data.py
```

With dependencies and the model also available locally, both generation and
scoring run without network access or an external judge:

```bash
aethereval --model /path/to/policy --tasks mbpp-plus --output-dir outputs
```

## Protocol

- Prompt uses the same short-reasoning-then-fenced-code format as our HumanEval+
  task, with `### Question`, `### Format`, and `### Answer` sections. The question
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
  Reference execution also honors `MBPP_OUTPUT_NOT_NONE_TASKS`.
- Primary `pass@1` and `accuracy_plus` require both base and Plus tests to pass.
  `accuracy_base` is reported separately on the same 378 tasks, not the original
  full MBPP benchmark. A base failure skips Plus execution without changing pass/fail.
- No runtime dataset downloads: scoring reads raw local inputs and computes
  reference outputs locally, cached per process.

Execute generated programs in an isolated environment without credentials or
valuable files. EvalPlus's reliability guard is not a security sandbox.
