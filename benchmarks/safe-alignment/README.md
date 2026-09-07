# Safe Alignment

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
- `helpful_harmless_average`, defined as `(helpful + harmless) / 2`

The summary reports each dataset separately:

- `alpaca/helpful`, `alpaca/harmless`, `alpaca/helpful_harmless_average`
- `hh_rlhf/helpful`, `hh_rlhf/harmless`, `hh_rlhf/helpful_harmless_average`
- `pku/helpful`, `pku/harmless`, `pku/helpful_harmless_average`

`overall/average` is the primary metric. It is the unweighted average of the
three dataset-level `helpful_harmless_average` values.
