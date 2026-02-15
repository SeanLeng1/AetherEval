from __future__ import annotations

import argparse
import json

from .config import load_yaml_config, resolve_run_arguments
from .core.runner import inspect_prompts, run_evaluation
from .core.task_register import list_task_default_gens, list_tasks


def _info(message: str) -> None:
    print(f"[aethereval] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AetherEval: lightweight generative-only vLLM eval framework."
    )
    parser.add_argument("--list-tasks", action="store_true", help="List discovered tasks and exit.")
    parser.add_argument(
        "--list-task-defaults",
        action="store_true",
        help="Print effective DEFAULT_GEN for all tasks and exit.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file path.")
    parser.add_argument("--tasks", type=str, default=None, help="Task names: all or comma-separated.")
    parser.add_argument("--model", type=str, default=None, help="Model name/path for vLLM.")
    parser.add_argument(
        "--inspect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print first 5 prompts after chat-template rendering and exit (no inference).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Default: <model_suffix_lower>.",
    )

    parser.add_argument("--dp-size", type=int, default=None, help="Data parallel worker count.")
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size per worker.",
    )

    parser.add_argument("--n", type=int, default=None, help="Override number of generations per sample.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override top-p.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k.")
    parser.add_argument("--min-p", type=float, default=None, help="Override min-p.")
    parser.add_argument("--seed", type=int, default=None, help="Override sampling seed.")

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

    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM model kwarg.")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM model kwarg.")
    parser.add_argument("--dtype", type=str, default=None, help="vLLM model kwarg.")
    parser.add_argument(
        "--vllm-arg",
        action="append",
        default=None,
        help="Extra vLLM model kwargs (repeatable), format: key=value",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing predictions.jsonl for the same run_id.",
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

    cfg = load_yaml_config(args.config)
    resolved = resolve_run_arguments(args, cfg)

    if not resolved["model"]:
        parser.error("--model is required unless --list-tasks is set.")

    _info(f"config={args.config if args.config else '(none)'}")
    _info(
        f"model={resolved['model']} tasks={resolved['tasks']} "
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
    if resolved["model_kwargs"]:
        _info(f"vllm_model_kwargs={resolved['model_kwargs']}")

    if resolved["inspect"]:
        inspected = inspect_prompts(
            model=resolved["model"],
            tasks=resolved["tasks"],
            model_kwargs=resolved["model_kwargs"],
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
        overwrite=resolved["overwrite"],
        run_id=resolved["run_id"],
        model_kwargs=resolved["model_kwargs"],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
