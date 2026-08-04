import argparse
import json
from pathlib import Path
from typing import Any

from aethereval.config import parse_key_value_args
from aethereval.core.task_defaults import (
    resolve_task_default_gen,
    resolve_task_num_repeats,
)

from .external import DEFAULT_CATEGORIES, ExternalRunSpec


def add_bfcl_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("BFCL")
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
            "BFCL request concurrency; defaults to max(16, 16 * dp_size), "
            "capped at 100."
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
        "--bfcl-context-length",
        type=int,
        default=None,
        help=(
            "BFCL-only context length; overrides the global backend setting after "
            "native tasks finish."
        ),
    )
    group.add_argument(
        "--bfcl-sglang-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Extra BFCL-only SGLang server argument; repeat as needed. Overrides "
            "the matching global --sglang-arg."
        ),
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


def _generation_value(
    overrides: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    value = overrides.get(key)
    return value if value is not None else defaults.get(key, fallback)


def _resolve_num_repeats(
    resolved: dict[str, Any],
    output_dir: Path,
) -> int:
    runtime_override = resolved.get("num_repeats")
    if not resolved.get("eval_only"):
        return resolve_task_num_repeats("bfcl", runtime_override)

    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return resolve_task_num_repeats("bfcl", runtime_override)
    with summary_path.open("r", encoding="utf-8") as file:
        saved_summary = json.load(file)
    saved_repeats = saved_summary.get("num_repeats")
    if saved_repeats is None:
        return resolve_task_num_repeats("bfcl", runtime_override)
    saved_repeats = int(saved_repeats)
    if runtime_override is not None and int(runtime_override) != saved_repeats:
        raise ValueError(
            "BFCL eval-only num_repeats conflicts with the saved summary: "
            f"requested={runtime_override}, saved={saved_repeats}."
        )
    return resolve_task_num_repeats("bfcl", saved_repeats)


def build_bfcl_spec(
    args: argparse.Namespace,
    resolved: dict[str, Any],
    output_dir: Path,
) -> ExternalRunSpec:
    backend = str(resolved["backend"])
    dp_size = int(resolved["dp_size"])
    tp_size = int(resolved["tp_size"])
    backend_kwargs = dict(resolved["backend_kwargs"])
    backend_kwargs.update(
        parse_key_value_args(args.bfcl_sglang_arg, "--bfcl-sglang-arg")
    )

    defaults = resolve_task_default_gen("bfcl", {})
    generation = resolved["gen_overrides"]
    n = int(_generation_value(generation, defaults, "n", 1))
    if n != 1:
        raise ValueError(
            "BFCL supports exactly one generation per test interaction (n=1); "
            "use --num-repeats for independent full benchmark runs."
        )
    context_length = args.bfcl_context_length
    if context_length is None:
        context_length = backend_kwargs.get(
            "context_length",
            backend_kwargs.get("max_model_len"),
        )
    elif backend == "sglang":
        backend_kwargs["context_length"] = context_length

    memory_fraction = (
        backend_kwargs.get("mem_fraction_static")
        if backend == "sglang"
        else backend_kwargs.get("gpu_memory_utilization")
    )
    return ExternalRunSpec(
        model=str(resolved["model"]),
        model_name=resolved["model_name"],
        output_dir=Path(output_dir),
        categories=_split_categories(args.categories),
        backend=backend,
        dp_size=dp_size,
        tp_size=tp_size,
        router_policy=args.bfcl_router_policy or "cache_aware",
        num_threads=(
            args.num_threads
            if args.num_threads is not None
            else min(100, max(16, 16 * dp_size))
        ),
        gpu_memory_utilization=float(
            memory_fraction if memory_fraction is not None else 0.9
        ),
        dtype=str(backend_kwargs.get("dtype", "bfloat16")),
        sglang_server_args=backend_kwargs if backend == "sglang" else {},
        temperature=float(
            _generation_value(generation, defaults, "temperature", 0.001)
        ),
        max_tokens=int(_generation_value(generation, defaults, "max_new_tokens", 4096)),
        max_context_length=context_length,
        top_p=float(_generation_value(generation, defaults, "top_p", 1.0)),
        top_k=int(_generation_value(generation, defaults, "top_k", -1)),
        seed=generation.get("seed"),
        num_repeats=_resolve_num_repeats(resolved, output_dir),
        verbose=bool(args.bfcl_verbose),
        allow_overwrite=bool(resolved["overwrite"]),
        run_generation=not resolved["eval_only"],
        run_evaluation=not resolved["generate_only"],
    )
