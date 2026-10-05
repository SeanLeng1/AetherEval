# LongBench-Write

## Official source and protocol

Audited 2026-10-05. Source: [THUDM/LongWriter](https://github.com/THUDM/LongWriter),
commit `447539b356a8b09760b51eca876e19b6fc1f2dd7`.

We use the full bilingual **120-prompt** `evaluation/longbench_write.jsonl`,
not the English-only file or LongWrite-Ruler. Source prompt, type and required
word count are retained. Language labels come from the official English subset,
not a character heuristic (one English prompt quotes Chinese text).
`n=1, max_new_tokens=32768, temperature=0.5` follows `evaluation/pred.py`;
upstream leaves top_p model-dependent, whereas AetherEval explicitly uses 1.

We retain the official `evaluation/judge.txt` prompt and six integer 1–5
dimensions: Relevance, Accuracy, Coherence, Clarity, Breadth and Depth, Reading
Experience. Quality `S_q` is `(mean of the six scores - 1) * 25`. The default
judge is `gpt-4o-2024-05-13`, temperature 0.5, max new tokens 1024, as in
`eval_quality.py`. An override/local judge changes the judging model and must
be labeled; such scores are not directly official-leaderboard-comparable.

Word counting exactly matches `pred.py`: count each CJK ideograph U+4E00–U+9FFF
and each ASCII alphabetic word. The required and generated counts feed the
piecewise formula in `eval_length.py`: for over-length output,
`S_l = 100 max(0, 1 - (actual/requested - 1)/3)`; otherwise
`S_l = 100 max(0, 1 - (requested/actual - 1)/2)`. Empty output receives 0
instead of the upstream division-by-zero error.

We report `quality_score` (`S_q`), `length_score` (`S_l`), each quality dimension,
language breakdowns, and `overall_score = (S_q + S_l)/2`. Judge failures are
marked unscored, make evaluation incomplete, and retry on resume; no invented
zero-quality score enters the reported averages. We use AetherEval's bounded
format retries (three normal attempts plus one structured local attempt), not
upstream's nested HTTP/format retry loop.

The common final-answer extraction strips text before the last `</think>`.
Quality and word-count scoring both see that same answer. The quality prompt
explicitly tells the judge not to score whether the requested word count was met;
that is assessed separately by `S_l`.

## Prepare and run

```bash
python benchmarks/init.py longbench-write
aethereval --model /path/to/model --tasks longbench-write \
  --judge-backend local --judge-models /path/to/gemma,/path/to/qwen \
  --no-judge-enable-thinking
```

Allow enough candidate and judge context for long responses. A global
`--max-new-tokens` override below 32768 changes the official output budget;
report that adaptation rather than silently treating it as protocol-identical.
