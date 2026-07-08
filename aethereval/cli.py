import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml_config, resolve_run_arguments
from .core.runner import inspect_prompts, run_evaluation
from .core.task_register import list_task_default_gens, list_tasks

EXTERNAL_BENCHMARKS = ("bfcl",)


def _info(message: str) -> None:
    print(f"[aethereval] {message}")


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("comma-separated argument cannot be empty")
    return items


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_external_common(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model is required with --external-benchmark.")
    if not args.output_dir:
        raise ValueError("--output-dir is required with --external-benchmark.")


def _build_external_spec(args: argparse.Namespace) -> tuple[Any, Any]:
    _require_external_common(args)
    backend = args.backend or "sglang"

    if args.external_benchmark == "bfcl":
        from benchmarks.bfcl.external import ExternalRunSpec, run

        num_gpus = (
            args.num_gpus
            if args.num_gpus is not None
            else args.tp_size
            if args.tp_size is not None
            else 1
        )
        spec = ExternalRunSpec(
            model=args.model,
            output_dir=Path(args.output_dir),
            model_path=args.model_path,
            categories=_split_csv(args.categories, ["all"]),
            backend=backend,
            num_gpus=num_gpus,
            gpu_memory_utilization=(
                args.gpu_memory_utilization
                if args.gpu_memory_utilization is not None
                else 0.9
            ),
            temperature=args.temperature if args.temperature is not None else 0.001,
            allow_overwrite=True if args.overwrite is None else bool(args.overwrite),
            run_generation=not args.skip_generation,
            run_evaluation=not args.skip_evaluation,
        )
        return spec, run

    raise ValueError(f"Unknown external benchmark: {args.external_benchmark}")


def run_external_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    spec, run = _build_external_spec(args)
    _info(
        f"external_benchmark={args.external_benchmark} model={args.model} "
        f"backend={spec.backend} output_dir={spec.output_dir}"
    )
    return _jsonable(run(spec))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AetherEval: lightweight generative-only LLM eval framework."
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="List discovered tasks and exit."
    )
    parser.add_argument(
        "--list-task-defaults",
        action="store_true",
        help="Print effective DEFAULT_GEN for all tasks and exit.",
    )
    parser.add_argument(
        "--external-benchmark",
        choices=EXTERNAL_BENCHMARKS,
        default=None,
        help="Run an external benchmark through AetherEval instead of native tasks.",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="YAML config file path."
    )
    parser.add_argument(
        "--tasks", type=str, default=None, help="Task names: all or comma-separated."
    )
    parser.add_argument("--model", type=str, default=None, help="Model name/path.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local checkpoint path for external benchmarks.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("vllm", "sglang"),
        default=None,
        help="Inference backend.",
    )
    parser.add_argument(
        "--inspect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print first 5 prompts after chat-template rendering and exit (no inference).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output root directory."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Default: <model_suffix_lower>.",
    )

    parser.add_argument(
        "--dp-size", type=int, default=None, help="Data parallel worker count."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size per worker.",
    )

    parser.add_argument(
        "--n", type=int, default=None, help="Override number of generations per sample."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Override max new tokens."
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override temperature."
    )
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p.")
    parser.add_argument(
        "--top-k", type=int, default=None, help="Override top-k (default: -1)."
    )
    parser.add_argument("--min-p", type=float, default=None, help="Override min-p.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override sampling seed."
    )

    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=None,
        help="Bootstrap resample count forwarded to benchmark metrics.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Bootstrap RNG seed forwarded to benchmark metrics.",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=None,
        help="Bootstrap confidence level in [0,1], forwarded to benchmark metrics.",
    )
    parser.add_argument(
        "--rm-model-path",
        type=str,
        default=None,
        help="Reward-model path forwarded to RM-based benchmark metrics.",
    )
    parser.add_argument(
        "--cm-model-path",
        type=str,
        default=None,
        help="Cost/safety-model path forwarded to RM-based benchmark metrics.",
    )
    parser.add_argument(
        "--rm-batch-size",
        type=int,
        default=None,
        help="Batch size for RM-based benchmark metrics.",
    )
    parser.add_argument(
        "--rm-max-length",
        type=int,
        default=None,
        help="Maximum token length for RM-based benchmark metrics.",
    )
    parser.add_argument(
        "--rm-device",
        type=str,
        default=None,
        help="Device for RM-based benchmark metrics, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--rm-dtype",
        type=str,
        default=None,
        help="Torch dtype for RM-based benchmark metrics, e.g. bfloat16.",
    )
    parser.add_argument(
        "--rm-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward trust_remote_code to RM tokenizers/models.",
    )

    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=None, help="vLLM model kwarg."
    )
    parser.add_argument(
        "--max-model-len", type=int, default=None, help="vLLM model kwarg."
    )
    parser.add_argument(
        "--mem-fraction-static", type=float, default=None, help="SGLang Engine kwarg."
    )
    parser.add_argument(
        "--context-length", type=int, default=None, help="SGLang Engine kwarg."
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="BFCL categories/collections for --external-benchmark bfcl, comma-separated.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="BFCL GPU count; defaults to --tp-size when provided.",
    )
    parser.add_argument(
        "--sglang-generation-batch-size",
        type=int,
        default=None,
        help="AetherEval SGLang generation batch size for progress updates.",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Backend model kwarg.")
    parser.add_argument(
        "--vllm-arg",
        action="append",
        default=None,
        help="Extra vLLM model kwargs (repeatable), format: key=value",
    )
    parser.add_argument(
        "--sglang-arg",
        action="append",
        default=None,
        help="Extra SGLang Engine kwargs (repeatable), format: key=value",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing predictions.jsonl for the same run_id.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="External benchmarks: reuse existing raw generations.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="External benchmarks: generate only and skip scoring.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tasks:
        for task_name in list_tasks():
            print(task_name)
        return
    if args.list_task_defaults:
        payload = list_task_default_gens()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if args.external_benchmark:
        try:
            result = run_external_benchmark(args)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    cfg = load_yaml_config(args.config)
    resolved = resolve_run_arguments(args, cfg)

    if not resolved["model"]:
        parser.error("--model is required unless --list-tasks is set.")

    _info(f"config={args.config if args.config else '(none)'}")
    _info(
        f"model={resolved['model']} backend={resolved['backend']} tasks={resolved['tasks']} "
        f"dp_size={resolved['dp_size']} tp_size={resolved['tp_size']} "
        f"overwrite={resolved['overwrite']}"
    )
    _info(
        f"output_dir={resolved['output_dir']} "
        f"run_id={resolved['run_id'] if resolved['run_id'] else '(auto:model_suffix)'}"
    )
    explicit_gen_overrides = {
        k: v for k, v in resolved["gen_overrides"].items() if v is not None
    }
    if explicit_gen_overrides:
        _info(f"generation_overrides={explicit_gen_overrides}")
    if resolved["backend_kwargs"]:
        _info(f"backend_model_kwargs={resolved['backend_kwargs']}")

    if resolved["inspect"]:
        inspected = inspect_prompts(
            model=resolved["model"],
            tasks=resolved["tasks"],
            model_kwargs=resolved["backend_kwargs"],
        )
        for task_name in inspected["tasks"]:
            print(f"=== {task_name} ===")
            rows = inspected["results"].get(task_name, [])
            if not rows:
                print("(no samples)")
                continue
            for idx, row in enumerate(rows, start=1):
                print(f"[{idx}] sample_id={row['sample_id']}")
                print(row["prompt"])
                if idx < len(rows):
                    print()
        return

    result = run_evaluation(
        model=resolved["model"],
        tasks=resolved["tasks"],
        output_dir=resolved["output_dir"],
        dp_size=resolved["dp_size"],
        tensor_parallel_size=resolved["tp_size"],
        gen_overrides=resolved["gen_overrides"],
        bootstrap_resamples=resolved["bootstrap_resamples"],
        bootstrap_seed=resolved["bootstrap_seed"],
        bootstrap_confidence=resolved["bootstrap_confidence"],
        metric_options=resolved["metric_options"],
        overwrite=resolved["overwrite"],
        run_id=resolved["run_id"],
        backend_name=resolved["backend"],
        backend_kwargs=resolved["backend_kwargs"],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
