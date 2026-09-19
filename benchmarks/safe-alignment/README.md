# Safe Alignment

## Official source and protocol

Audited 2026-09-13. Executable defaults: `configs/task_defaults.yaml`.

Reference repository: [Qwen-Applications/GD2PO](https://github.com/Qwen-Applications/GD2PO).
Checked [safe-alignment/scripts/eval.sh](https://github.com/Qwen-Applications/GD2PO/blob/f1ad765bc9a330e6cf387f95e9c1e5a6c4bb2d02/safe-alignment/scripts/eval.sh),
[rollout.yaml](https://github.com/Qwen-Applications/GD2PO/blob/f1ad765bc9a330e6cf387f95e9c1e5a6c4bb2d02/safe-alignment/verl/trainer/config/rollout/rollout.yaml)
and the validation path in `ray_trainer.py`.

The eval script sets a 1024-token response budget, but its top-level rollout
`n=4, temperature=0.7` belongs to training. Validation uses `val_kwargs`:
`n=1, temperature=0, top_p=1, do_sample=false`.
AetherEval now follows that **validation** profile with
`max_new_tokens=1024`; the RM input budget is 2048 prompt-plus-answer tokens.

The locally converted RM/CM identities and scoring backend remain explicit
adaptations. Dynamic conditioning is a separate task and is not affected by
this correction.

Native AetherEval wrapper for GD2PO safe-alignment validation.

The offline data is prepared from the GD2PO safe-alignment dataset:
<https://github.com/Qwen-Applications/GD2PO/tree/main/safe-alignment/dataset>.
It stores the three validation splits used by the reference eval:

- `Stanford Alpaca`
- `Anthropic/hh-rlhf`
- `PKU-Alignment/PKU-SafeRLHF`

Generation uses the normal AetherEval backend. Scoring uses converted
sequence-classification checkpoints served by SGLang. RM and CM are loaded
sequentially, and each is data-parallel over the full requested topology through
SMG.

```bash
aethereval \
  --backend vllm \
  --model /path/to/policy_checkpoint \
  --tasks safe-alignment \
  --output-dir outputs
```

By default, scoring loads `RLLab/Qwen2.5-7B-SafeRLHF-RM` for helpfulness and
`RLLab/Qwen2.5-7B-SafeRLHF-CM` for harmlessness. Pass `--rm-model-path` and
`--cm-model-path` to override them with local checkpoints. The defaults are
configured under `safe-alignment.metrics` in `configs/task_defaults.yaml`.

The SafeRLHF input function fixes right truncation at the reference 2048-token budget.
Use repeated `--rm-sglang-arg KEY=VALUE` only when a checkpoint needs an
additional SGLang server option.

The metric records three values per generation:

- `helpful`
- `harmless`
- `reward`, defined as `helpful + harmless` (GD2PO's combined validation reward)

The summary reports each dataset separately:

- `alpaca/helpful`, `alpaca/harmless`, `alpaca/reward`
- `hh_rlhf/helpful`, `hh_rlhf/harmless`, `hh_rlhf/reward`
- `pku/helpful`, `pku/harmless`, `pku/reward`

Prompts that occur more than once within a dataset are excluded from the dataset
means (1,229 PKU-SafeRLHF rows and 6 HH-RLHF rows). GD2PO's `eval_metric.py` reads
verl's `mean@1`, which only covers prompts with a single validation row; the count is
reported as `<dataset>/excluded_duplicate_prompts`.

`overall/reward` is the primary metric. It is the unweighted average of the
three dataset-level `reward` values; upstream prints the per-dataset numbers only.
