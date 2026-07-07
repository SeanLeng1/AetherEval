"""CLI for the API-Bank (GD2PO) external benchmark.

    # base HF model:
    python -m benchmarks.apibank --model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir outputs/apibank-base
    # local checkpoint (any name + --model-path):
    python -m benchmarks.apibank --model rlla-gdpo --model-path /ckpt \
        --output-dir outputs/apibank-gdpo

Defaults to the sglang backend (the tmux0 container ships sglang, not vllm).
"""

import argparse
from pathlib import Path

from .external import ExternalRunSpec, run


def _fmt(value, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser(description="API-Bank (GD2PO) eval for ToolRL/GDPO models")
    ap.add_argument("--model", required=True,
                    help="HF id (base) or any name paired with --model-path")
    ap.add_argument("--model-path", default=None, help="local checkpoint dir (overrides HF load)")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--levels", default="1,2,3", help="comma-separated API-Bank levels")
    ap.add_argument("--backend", default="sglang", choices=["sglang", "vllm"])
    ap.add_argument("--dp-size", type=int, default=1)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.6, help="vllm engine arg")
    ap.add_argument("--mem-fraction-static", type=float, default=0.8, help="sglang engine arg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--skip-generation", action="store_true", help="only re-score existing result.json")
    ap.add_argument("--skip-evaluation", action="store_true", help="only generate")
    args = ap.parse_args()

    spec = ExternalRunSpec(
        model=args.model,
        output_dir=args.output_dir,
        model_path=args.model_path,
        levels=[lv.strip() for lv in args.levels.split(",") if lv.strip()],
        backend=args.backend,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mem_fraction_static=args.mem_fraction_static,
        seed=args.seed,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        run_generation=not args.skip_generation,
        run_evaluation=not args.skip_evaluation,
    )
    result = run(spec)
    if result.metrics:
        m = result.metrics
        print("\n=== API-Bank (GD2PO) metrics ===")
        print(f"  acc={_fmt(m['overall_acc'], '%')} "
              f"(lv1={_fmt(m['lv1_acc'], '%')}, lv2={_fmt(m['lv2_acc'], '%')}, lv3={_fmt(m['lv3_acc'], '%')})")
        print(f"  format={_fmt(m['overall_format_acc'], '%')} "
              f"(lv1={_fmt(m['format_lv1_acc'], '%')}, lv2={_fmt(m['format_lv2_acc'], '%')}, "
              f"lv3={_fmt(m['format_lv3_acc'], '%')})")
        print(f"  length={_fmt(m['overall_length_avg'])} "
              f"(lv1={_fmt(m['length_avg_lv1'])}, lv2={_fmt(m['length_avg_lv2'])}, "
              f"lv3={_fmt(m['length_avg_lv3'])})")
        print(f"  primary: {result.primary_metric} = {result.primary_score:.2f}")
        print(f"summary -> {Path(spec.output_dir).resolve() / 'summary.json'}")


if __name__ == "__main__":
    main()
