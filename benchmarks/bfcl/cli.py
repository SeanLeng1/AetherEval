import argparse
import json
from pathlib import Path
from typing import Any, Callable

from aethereval.core.task_defaults import (
    resolve_phase_num_repeats,
    resolve_task_default_gen,
)

from .external import DEFAULT_CATEGORIES, ExternalRunSpec, run
from .register import HANDLER_PROFILES


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("BFCL")
    group.add_argument(
        "--bfcl-handler",
        choices=HANDLER_PROFILES,
        default=None,
        help=(
            "BFCL output protocol: toolrl for <tool_call>/<response> models "
            "(default), or official for an exact prompt-mode model registered "
            "by the installed BFCL V3 package."
        ),
    )
    group.add_argument(
        "--categories",
        default=None,
        help=(
            "BFCL categories/collections, comma-separated "
            "(default: live,non_live,multi_turn; use all for full V3)."
        ),
    )
    group.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help=(
            "BFCL request concurrency; defaults to max(64, 64 * dp_size), "
            "matching native SGLang tasks. Pass 100 to reproduce runs made "
            "before this default changed."
        ),
    )
    group.add_argument(
        "--bfcl-router-policy",
        choices=(
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ),
        default=None,
        help="SGLang Model Gateway routing policy (default: cache_aware).",
    )
    group.add_argument(
        "--bfcl-verbose",
        action="store_true",
        help="Show verbose BFCL multi-turn step logs.",
    )


def _split_categories(value: str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_CATEGORIES)
    categories = [item.strip() for item in value.split(",") if item.strip()]
    if not categories:
        raise ValueError("--categories cannot be empty")
    return categories


def _cast_if_set(cast: Callable[[Any], Any], value: Any) -> Any:
    return None if value is None else cast(value)


def build_bfcl_spec(
    args: argparse.Namespace,
    resolved: dict[str, Any],
    output_dir: Path,
) -> ExternalRunSpec:
    backend = str(resolved["backend"])
    dp_size = int(resolved["dp_size"])
    backend_kwargs = dict(resolved["backend_kwargs"])

    generation = resolved["gen_overrides"]
    n = generation.get("n")
    if n is None:
        n = resolve_task_default_gen("bfcl").get("n", 1)
    if int(n) != 1:
        raise ValueError(
            "BFCL supports exactly one generation per test interaction (n=1); "
            "use --num-repeats for independent full benchmark runs."
        )

    # Only explicit settings are passed; ExternalRunSpec holds the defaults.
    memory_fraction = backend_kwargs.get(
        "mem_fraction_static" if backend == "sglang" else "gpu_memory_utilization"
    )
    overrides = {
        "handler": args.bfcl_handler,
        "router_policy": args.bfcl_router_policy,
        "gpu_memory_utilization": _cast_if_set(float, memory_fraction),
        "dtype": _cast_if_set(str, backend_kwargs.get("dtype")),
        "temperature": _cast_if_set(float, generation.get("temperature")),
        "max_tokens": _cast_if_set(int, generation.get("max_new_tokens")),
        "top_p": _cast_if_set(float, generation.get("top_p")),
        "top_k": _cast_if_set(int, generation.get("top_k")),
    }
    return ExternalRunSpec(
        model=str(resolved["model"]),
        model_name=resolved["model_name"],
        output_dir=Path(output_dir),
        categories=_split_categories(args.categories),
        backend=backend,
        dp_size=dp_size,
        tp_size=int(resolved["tp_size"]),
        num_threads=(
            args.num_threads
            if args.num_threads is not None
            else max(64, 64 * dp_size)
        ),
        sglang_server_args=backend_kwargs if backend == "sglang" else {},
        max_context_length=backend_kwargs.get(
            "context_length", backend_kwargs.get("max_model_len")
        ),
        seed=generation.get("seed"),
        enable_thinking=generation.get("enable_thinking"),
        num_repeats=resolve_phase_num_repeats(
            "bfcl",
            Path(output_dir) / "summary.json",
            runtime_override=resolved["num_repeats"],
            eval_only=bool(resolved["eval_only"]),
        ),
        verbose=bool(args.bfcl_verbose),
        allow_overwrite=bool(resolved["overwrite"]),
        run_generation=not resolved["eval_only"],
        run_evaluation=not resolved["generate_only"],
        **{key: value for key, value in overrides.items() if value is not None},
    )


def run_external(
    args: argparse.Namespace,
    resolved: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run BFCL for the AetherEval CLI and return its task summary."""
    spec = build_bfcl_spec(args, resolved, output_dir)
    print(
        f"[aethereval] external_task=bfcl model={spec.model} "
        f"backend={spec.backend} dp_size={spec.dp_size} tp_size={spec.tp_size} "
        f"num_repeats={spec.num_repeats} output_dir={spec.output_dir}"
    )
    run(spec)
    with (Path(output_dir) / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["task"] = "bfcl"
    return summary
