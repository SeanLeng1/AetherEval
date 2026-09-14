# WritingBench

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official repository: [X-PLUG/WritingBench](https://github.com/X-PLUG/WritingBench).
Checked the [leaderboard recipe](https://github.com/X-PLUG/WritingBench/blob/ae2d5176449b7b769815482641d35926f26793eb/README.md)
and `evaluator/critic.py`.

Candidate defaults follow the leaderboard:
`n=1, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=16000`.
Use the model's actual lower limit if it cannot support that budget.
The earlier 8192 cap was a local small-model profile, not the official default.
Judge defaults remain `claude-sonnet-4-5, temperature=1, top_p=0.95,
max_new_tokens=2048`, matching the published November 2025 judge update.
Model context capacity must accommodate both the prompt and output budget;
16000 is an output limit, not a total context length.

Native implementation of the 1,000-item WritingBench release.

- Candidate generation: `n=1`, temperature `0.7`, top-p `0.8`, top-k `20`, max new tokens `16000` (or the model's actual lower limit).
- Judge: `claude-sonnet-4-5`, temperature `1.0`, top-p `0.95`, max tokens `2048`.
- Each of the five instance-specific criteria is judged independently once.
- `overall_score` is the upstream 1–10 mean scaled to 0–100.
- The current style/format/length requirement subset files are included; both
  response-level (`*_R`) and selected-criterion (`*_C`) metrics follow
  `calculate_scores.py` and are scaled to 0–100.

The implementation is native; online judge inference uses LiteLLM provider routing.
