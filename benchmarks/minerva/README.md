# Minerva Math Benchmark

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Official source: [Minerva paper](https://arxiv.org/html/2206.14858v2), particularly
Sections 2.3–2.5 and the OCWCourses appendix.
No standalone author-maintained OCWCourses inference repository was verified.
The local [RLLab/eval-set/minervamath](https://huggingface.co/datasets/RLLab/eval-set)
contains OCW-style problems; it is not the lm-evaluation-harness
`minerva_math` variant over the MATH dataset.

The paper uses a 512-token generation limit, greedy single-sample evaluation,
and `temperature=0.6, top_p=0.95` for multiple samples (64 for OCW majority
voting). AetherEval uses the shared math profile `n=16, temperature=0.6, top_p=0.95, max_new_tokens=32768`
instead of the 512-token path: the prompt asks for step-by-step reasoning, which 512 tokens truncate.
The local prompt and math-verify scorer still differ from the paper's few-shot
evaluation; this is not a reproduction of the paper's protocol.
Long-reasoning evaluations should explicitly override the output budget and report it.

## Data

- Source dataset: `RLLab/eval-set` (config: `minervamath`, split: `train`)
- Local offline file: `data/eval.jsonl`
- Regeneration script: `prepare_data.py`

Rows keep the source `problem` and `solution` fields. The full `solution` text is used
as the gold answer, with the five symbolic physics repairs described below.

## Metrics

- Primary metric: `accuracy`
- Scored with shared `math-verify` logic from `benchmark_utils/`, except for the
  five symbolic physics answers below
- A prediction that does not match is parsed again with unit stripping disabled, as
  Qwen2.5-Math's `skip_unit` does for Minerva. Otherwise trailing variables are dropped
  as units (`\frac{37}{4} m` becomes `37/4`, `\frac{t}{4}\sin 2t` loses its last `t`).
  Other math tasks keep stock `math-verify`.
- Reports `accuracy@n`, `pass@k`, and parsed-rate metrics when multiple generations are used

### Symbolic physics grading repairs

`math-verify` 0.9.0 cannot parse subscripted `E_{n}` (it reads `E` as Euler's number),
`X_{\odot}`, or some products containing `\gamma` (read as the Gamma function).
`symbolic.py` repairs these five rows during scoring:

| Row | Gold | Repair |
|---|---|---|
| `minervamath_27` | `\frac{dM}{dt}=\frac{10^{5} L_{\odot}}{...} M^{6}` | symbolic solar identifiers; compare the rate or an equivalent linear ODE |
| `minervamath_261` | `\hbar \omega(v+1/2)-\frac{E_{0}^{2} e^{2}}{2 m \omega^{2}}` | extract the final boxed reference; treat indexed energy and electric charge as variables |
| `minervamath_268` | `\frac{1}{3} E_{1}+\frac{2}{3} E_{2}` | symbolic indexed energies |
| `minervamath_269` | `E_{1},E_{2}` | symbolic indexed energies; compare the answer set |
| `minervamath_138` | contains `\gamma` | treat Lorentz gamma as a variable |

The grader takes the final boxed reference, applies the same temporary symbol
aliases to the reference and prediction, and compares finite SymPy expressions
with `math-verify`. It disables unit stripping for these physics expressions.
It does not fall back to string comparison or award credit for failed parsing.
The original data, prompts, golds and saved model outputs remain intact, and all
272 questions remain in the denominator. Regression cases cover algebraically
equivalent answers and wrong signs, coefficients, missing terms and extra energies.

Re-score saved predictions for **every compared model** with `--eval-only` after
updating the scorer. Generation does not need to be repeated. The five rows can
contribute up to 5/272 (1.84 percentage points); this is not a significance
threshold for comparisons between models.
