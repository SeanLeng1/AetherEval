# LiveCodeBench Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).
Checked [runner/parser.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/runner/parser.py)
and [vllm_runner.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/runner/vllm_runner.py).

Defaults retain the reference sampling settings with a local extended budget:
`n=10, temperature=0.2, top_p=0.95, max_new_tokens=32768`, reporting pass@1/5/10.
The 32768-token ceiling includes reasoning and code, replacing the reference
runner's 2000-token default. Report this budget explicitly; it is not the
unmodified reference profile. The serving context must also fit the prompt.
The reference `stop="###"` is not used: upstream applies it to base-model
completion only, and under this task's reasoning prompt it would truncate the
answer at the first markdown heading. Candidate tests run in a spawned worker
with numpy preloaded (as the reference harness does) under a 4 GiB memory limit.

The local `lighteval/code_generation_lite` v6 snapshot is not automatically the
official runner's `release_latest` set. Prompt formatting, date window and local
execution limits must also match before comparing against a leaderboard score.

## Files

```text
benchmarks/livecodebench/
  README.md
  data/eval.jsonl
  task.py
  metrics.py
  lcb_eval_runtime.py
  prepare_data.py
```

## Data

- Source dataset: `lighteval/code_generation_lite` (split: `test`)
- Source subset: `v6`
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Why `v6`:
- This task tracks the latest version-window benchmark (`v6`) and keeps local data size manageable for offline usage.
- As checked on `2026-02-14`, `v6` has `175` rows.

`prepare_data.py` stores compact raw test fields (`public_test_cases` + encoded `private_test_cases`).
`task.py` decodes tests during sample loading and then evaluates fully offline.

## Prompting

- Implemented in `task.py`
- Shared with HumanEval+, MBPP+ and LiveCodeBench in `benchmark_utils/code_prompt.py`.
  It is the AetherRL training prompt (`data/process_code.py` `code_prompt`) verbatim:
  a single user turn with no system message:
  `### Question:` + the question, then `### Format:` + "Please think step by step, then
  write the complete solution." + an interface line + "Put the final solution in one
  Python code block:" and a ```` ```python ```` block.
- This is not the official generic template
  ([prompts/code_generation.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/prompts/code_generation.py),
  `get_generic_question_template_answer` with `SYSTEM_MESSAGE_GENERIC`); the Question/Format
  sections are the same, but there is no system message, no `### Answer` line, and the
  format asks for step-by-step reasoning.
  - with starter code: "Return the completed Python function(s), preserving the requested
    names and signatures. The tested callable is `<fn_name>`."; the code block shows the
    starter code, as in the official template
  - without starter code: "Read input from stdin and write the answer to stdout; do not
    hard-code the examples."; the code block holds `# YOUR CODE HERE`
- The framework applies the model chat template.

## Metrics

- Implemented in `metrics.py`
- Runtime executor is benchmark-local in `lcb_eval_runtime.py` (call-based + stdio execution).
- Each candidate runs in a fresh subprocess with a 4 GiB address-space/data limit.
  Python allocation failures are reported as `Memory Limit Exceeded` and count as
  failed answers, not excluded samples. This is a local execution limit, not a
  claim about the official leaderboard's memory budget. Parallel workers each
  have this limit; it is not a whole-job memory cap.
- As in official LCB, candidates may convert integers of up to 50,000 digits.
  The limit is raised only in the candidate process, so it does not apply to
  other tasks scored in the same run.
- Known leniency, kept to preserve scores: the official `timeout_handler`
  (`testing_util.py` at `28fef95`) prints `timeout occured: alarm went off`
  before raising, and stdio grading captures that line; ours raises silently.
  A stdio candidate that catches the timeout (for example `except Exception:`)
  and then prints the right answer therefore passes here (`[True]`) but gets
  Wrong Answer (`[-2]`) from the official grader. Matching upstream would be a
  score change.
- Code extraction uses fenced-code parsing (last fenced block; no raw-text fallback).
- Per generation score:
  - `1.0` if all tests pass
  - `0.0` otherwise

Reported metrics:
- `accuracy`, `accuracy_stderr`
- `accuracy@n` when `n>1`
- `pass@k` (`k=1,2,4,...,n` by default)
- `parsed_rate`
- `accuracy_<platform>` (e.g., `accuracy_atcoder`, `accuracy_leetcode`)
