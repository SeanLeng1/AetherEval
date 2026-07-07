"""CLI for the BFCL-v3 external benchmark.

    # base HF model (registry name = HF id):
    python -m benchmarks.bfcl --model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir outputs/bfcl-base --categories non_live
    # local checkpoint (any registry name + --model-path):
    python -m benchmarks.bfcl --model rlla-gdpo --model-path /ckpt \
        --output-dir outputs/bfcl-gdpo

Defaults to the sglang backend (the tmux0 container ships sglang, not vllm).
"""

import argparse
from pathlib import Path

from .external import ExternalRunSpec, run


def main() -> None:
    ap = argparse.ArgumentParser(description="BFCL-v3 eval for ToolRL/GDPO models")
    ap.add_argument("--model", required=True,
                    help="registry name = HF id (base) or any name paired with --model-path")
    ap.add_argument("--model-path", default=None, help="local checkpoint dir (overrides HF load)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--categories", default="all",
                    help="comma-separated bfcl categories/collections (all|non_live|live|multi_turn|...)")
    ap.add_argument("--backend", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.001)
    ap.add_argument("--skip-generation", action="store_true", help="only re-evaluate existing results")
    ap.add_argument("--skip-evaluation", action="store_true", help="only generate")
    args = ap.parse_args()

    spec = ExternalRunSpec(
        model=args.model,
        output_dir=args.output_dir,
        model_path=args.model_path,
        categories=[c.strip() for c in args.categories.split(",") if c.strip()],
        backend=args.backend,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        run_generation=not args.skip_generation,
        run_evaluation=not args.skip_evaluation,
    )
    result = run(spec)
    print("\n=== BFCL-v3 metrics ===")
    for k, v in result.metrics.items():
        if not k.startswith("cat/"):
            print(f"  {k}: {v:.2f}")
    print(f"  primary: {result.primary_metric} = {result.primary_score:.2f}")
    print(f"summary -> {Path(spec.output_dir).resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
