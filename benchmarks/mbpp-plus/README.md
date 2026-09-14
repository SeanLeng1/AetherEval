# MBPP+

Pinned dataset: **MBPP+ v0.2.0, 378 tasks**, full release (not NoExtreme).
Pinned evaluator: **EvalPlus 0.3.1** (already a project dependency).

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

- Prompt matches the [EvalPlus chat API provider](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/provider/openai.py),
  using the instruction in [codegen.py](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/codegen.py).
  Includes the official coding-assistant system message and no additional reasoning
  instruction. The local backend applies
  the model's chat template. This is a chat profile, not EvalPlus's base-model
  completion or vLLM assistant-prefill profile.
- Greedy defaults: `n=1`, `temperature=0`, `top_p=0.95`, `max_new_tokens=768`.
  `top_p` follows the official chat request helper (temperature zero is greedy).
  The output budget is the reference decoder default; explicitly report overrides.
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
