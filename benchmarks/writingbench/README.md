# WritingBench

Native implementation of the 1,000-item WritingBench release.

- Candidate generation: `n=1`, temperature `0.7`, top-p `0.8`, top-k `20`, max new tokens `8192` for AetherEval's small-open-model protocol (the official WritingBench leaderboard uses `16000`).
- Judge: `claude-sonnet-4-5`, temperature `1.0`, top-p `0.95`, max tokens `2048`.
- Each of the five instance-specific criteria is judged independently once.
- `overall_score` is the upstream 1–10 mean scaled to 0–100.
- The current style/format/length requirement subset files are included; both
  response-level (`*_R`) and selected-criterion (`*_C`) metrics follow
  `calculate_scores.py` and are scaled to 0–100.

The implementation is native; online judge inference uses LiteLLM provider routing.
