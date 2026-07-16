import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml_config, resolve_run_arguments
from .core.io import ensure_dir, model_output_name, run_output_dir, write_json
from .core.runner import inspect_prompts, run_evaluation
from .core.task_defaults import resolve_task_default_gen
from .core.task_register import list_task_default_gens, list_tasks

EXTERNAL_TASKS = ("bfcl",)


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


def _parse_tasks_arg(tasks_arg: str) -> list[str]:
    selected = [item.strip() for item in tasks_arg.split(",") if item.strip()]
    if not selected:
        raise ValueError("No tasks selected.")
    return selected


def _split_native_external_tasks(tasks_arg: str) -> tuple[list[str], list[str]]:
    native_available = set(list_tasks())
    if tasks_arg.strip() == "all":
        return sorted(native_available), []

    selected = _parse_tasks_arg(tasks_arg)
    native_tasks: list[str] = []
    external_tasks: list[str] = []
    unknown: list[str] = []
    for task_name in selected:
        if task_name in native_available:
            native_tasks.append(task_name)
        elif task_name in EXTERNAL_TASKS:
            external_tasks.append(task_name)
        else:
            unknown.append(task_name)

    if unknown:
        available = sorted(native_available | set(EXTERNAL_TASKS))
        raise ValueError(
            f"Unknown tasks: {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(available)}"
        )
    return native_tasks, external_tasks


def _require_external_common(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model is required for external tasks.")


def _build_external_spec(
    args: argparse.Namespace,
    task_name: str,
    output_dir: Path | None = None,
) -> tuple[Any, Any]:
    _require_external_common(args)
    backend = args.backend or "sglang"
    effective_output_dir = Path(output_dir) if output_dir is not None else Path(
        args.output_dir or "outputs"
    )

    if task_name == "bfcl":
        from benchmarks.bfcl.external import ExternalRunSpec, run

        from .config import _parse_sglang_args

        default_gen = resolve_task_default_gen("bfcl", {})

        if backend == "sglang":
            tp_size = int(args.tp_size if args.tp_size is not None else 1)
            if args.num_gpus is not None and args.dp_size is None:
                if int(args.num_gpus) % tp_size:
                    raise ValueError("--num-gpus must be divisible by --tp-size")
                dp_size = int(args.num_gpus) // tp_size
            else:
                dp_size = int(args.dp_size if args.dp_size is not None else 1)
        else:
            dp_size = int(args.dp_size if args.dp_size is not None else 1)
            tp_size = int(
                args.tp_size
                if args.tp_size is not None
                else args.num_gpus
                if args.num_gpus is not None
                else 1
            )
        num_gpus = dp_size * tp_size
        if args.num_gpus is not None and int(args.num_gpus) != num_gpus:
            raise ValueError(
                "--num-gpus must equal --dp-size * --tp-size when combined"
            )

        use_sglang_router = (
            True
            if getattr(args, "bfcl_use_sglang_router", None) is None
            else bool(args.bfcl_use_sglang_router)
        )
        backend_kwargs = dict(getattr(args, "backend_kwargs", {}) or {})
        bfcl_sglang_kwargs = _parse_sglang_args(
            getattr(args, "bfcl_sglang_arg", None)
        )
        bfcl_backend_kwargs = {**backend_kwargs, **bfcl_sglang_kwargs}
        if backend == "sglang":
            bfcl_backend_kwargs.setdefault("log_level", "warning")
            if use_sglang_router and dp_size > 1:
                bfcl_backend_kwargs.setdefault("router_log_level", "warn")
        mem_fraction_static = backend_kwargs.get(
            "mem_fraction_static",
            getattr(args, "mem_fraction_static", None),
        )
        dtype = backend_kwargs.get("dtype", getattr(args, "dtype", None))
        default_num_threads = min(100, max(16, 16 * dp_size))
        spec = ExternalRunSpec(
            model=args.model,
            output_dir=effective_output_dir,
            model_name=getattr(args, "model_name", None),
            categories=_split_csv(args.categories, ["all"]),
            backend=backend,
            num_gpus=num_gpus,
            dp_size=dp_size,
            tp_size=tp_size,
            use_sglang_router=use_sglang_router,
            router_policy=(
                args.bfcl_router_policy
                if getattr(args, "bfcl_router_policy", None) is not None
                else "cache_aware"
            ),
            num_threads=(
                args.num_threads
                if args.num_threads is not None
                else default_num_threads
            ),
            gpu_memory_utilization=(
                mem_fraction_static
                if backend == "sglang" and mem_fraction_static is not None
                else args.gpu_memory_utilization
                if args.gpu_memory_utilization is not None
                else 0.9
            ),
            dtype=str(dtype if dtype is not None else "bfloat16"),
            sglang_server_args=(
                bfcl_backend_kwargs if backend == "sglang" else {}
            ),
            temperature=(
                args.temperature
                if args.temperature is not None
                else float(default_gen.get("temperature", 0.001))
            ),
            max_tokens=(
                args.max_new_tokens
                if args.max_new_tokens is not None
                else int(default_gen.get("max_new_tokens", 4096))
            ),
            max_context_length=(
                args.bfcl_context_length
                if getattr(args, "bfcl_context_length", None) is not None
                else args.context_length
                if args.context_length is not None
                else args.max_model_len
            ),
            top_p=(
                args.top_p
                if args.top_p is not None
                else float(default_gen.get("top_p", 1.0))
            ),
            top_k=(
                args.top_k
                if args.top_k is not None
                else int(default_gen.get("top_k", -1))
            ),
            seed=args.seed,
            verbose=bool(args.bfcl_verbose),
            allow_overwrite=True if args.overwrite is None else bool(args.overwrite),
            run_generation=not (
                args.skip_generation or getattr(args, "eval_only", False)
            ),
            run_evaluation=not (
                args.skip_evaluation or getattr(args, "generate_only", False)
            ),
        )
        return spec, run

    raise ValueError(f"Unknown external task: {task_name}")


def _mean_numeric_metrics(task_summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for summary in task_summaries.values():
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped.setdefault(str(key), []).append(float(value))
    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }


def _mean_primary_score(task_summaries: dict[str, dict[str, Any]]) -> float | None:
    values = [
        float(summary["primary_score"])
        for summary in task_summaries.values()
        if isinstance(summary.get("primary_score"), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _load_existing_task_summaries(
    run_root: Path,
    skip_tasks: set[str],
) -> dict[str, dict[str, Any]]:
    if not run_root.exists():
        return {}

    summaries: dict[str, dict[str, Any]] = {}
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name in skip_tasks:
            continue
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            raise ValueError(f"Existing summary must be a JSON object: {summary_path}")
        summaries[child.name] = summary
    return summaries


def _external_summary(task_name: str, task_output_dir: Path, result: Any) -> dict[str, Any]:
    summary_path = task_output_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            raise ValueError(f"External summary must be a JSON object: {summary_path}")
    else:
        summary = {}

    result_json = _jsonable(result)
    summary.setdefault("task", task_name)
    summary.setdefault("benchmark", task_name)
    summary.setdefault("external", True)
    summary.setdefault("metrics", result_json.get("metrics", {}))
    summary.setdefault("primary_metric", result_json.get("primary_metric"))
    summary.setdefault("primary_score", result_json.get("primary_score"))
    return summary


def _write_combined_run_summary(
    *,
    run_root: Path,
    run_id: str,
    selected_tasks: list[str],
    model: str,
    model_name: str,
    backend: str,
    phase: str,
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    primary_scores = {
        task_name: {
            "metric": summary.get("primary_metric"),
            "score": summary.get("primary_score"),
        }
        for task_name, summary in task_summaries.items()
    }
    run_summary = {
        "run_id": run_id,
        "selected_tasks": selected_tasks,
        "tasks": sorted(task_summaries.keys()),
        "model": model,
        "model_name": model_name,
        "backend": backend,
        "phase": phase,
        "results": task_summaries,
        "primary_scores": primary_scores,
        "primary_score_aggregate": _mean_primary_score(task_summaries),
        "summary": {
            "num_tasks": len(task_summaries),
            "metrics": _mean_numeric_metrics(task_summaries),
        },
    }
    write_json(run_root / "run_summary.json", run_summary)
    return run_summary


def run_selected_tasks(
    args: argparse.Namespace,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    if resolved["inspect"] and (
        resolved["generate_only"] or resolved["eval_only"]
    ):
        raise ValueError("--inspect cannot be combined with a phase-only mode")
    if resolved["generate_only"] and args.skip_generation:
        raise ValueError("--generate-only cannot be combined with --skip-generation")
    if resolved["eval_only"] and args.skip_evaluation:
        raise ValueError("--eval-only cannot be combined with --skip-evaluation")

    native_tasks, external_tasks = _split_native_external_tasks(resolved["tasks"])
    if not native_tasks and not external_tasks:
        raise ValueError("No tasks selected.")
    if resolved["inspect"] and external_tasks:
        raise ValueError(
            "--inspect is only supported for native tasks; external tasks requested: "
            f"{', '.join(external_tasks)}"
        )

    if resolved["inspect"]:
        inspected = inspect_prompts(
            model=resolved["model"],
            tasks=",".join(native_tasks),
            model_kwargs=resolved["backend_kwargs"],
            gen_overrides=resolved["gen_overrides"],
        )
        return {"inspect": inspected}

    effective_model_name = model_output_name(
        str(resolved["model"]), resolved["model_name"]
    )
    run_id = resolved["run_id"] or effective_model_name
    run_root = run_output_dir(
        resolved["output_dir"],
        str(resolved["model"]),
        resolved["run_id"],
        resolved["model_name"],
    )
    ensure_dir(run_root)

    native_result: dict[str, Any] | None = None
    if native_tasks:
        def run_native_phase(
            *,
            generate_only: bool,
            eval_only: bool,
            overwrite: bool,
        ) -> dict[str, Any]:
            return run_evaluation(
                model=resolved["model"],
                model_name=resolved["model_name"],
                tasks=",".join(native_tasks),
                output_dir=resolved["output_dir"],
                dp_size=resolved["dp_size"],
                tensor_parallel_size=resolved["tp_size"],
                gen_overrides=resolved["gen_overrides"],
                bootstrap_resamples=resolved["bootstrap_resamples"],
                bootstrap_seed=resolved["bootstrap_seed"],
                bootstrap_confidence=resolved["bootstrap_confidence"],
                metric_options=resolved["metric_options"],
                overwrite=overwrite,
                run_id=resolved["run_id"],
                backend_name=resolved["backend"],
                backend_kwargs=resolved["backend_kwargs"],
                generate_only=generate_only,
                eval_only=eval_only,
            )

        local_judge = (
            str(resolved["metric_options"].get("judge_backend", "api")).lower()
            == "local"
        )
        if (
            local_judge
            and not resolved["generate_only"]
            and not resolved["eval_only"]
        ):
            _info(
                "offline judge selected: generating candidates first, then "
                "restarting in eval-only mode so candidate and judge models do "
                "not share GPU memory"
            )
            run_native_phase(
                generate_only=True,
                eval_only=False,
                overwrite=resolved["overwrite"],
            )
            native_result = run_native_phase(
                generate_only=False,
                eval_only=True,
                overwrite=False,
            )
        else:
            native_result = run_native_phase(
                generate_only=resolved["generate_only"],
                eval_only=resolved["eval_only"],
                overwrite=resolved["overwrite"],
            )
        task_summaries = dict(native_result["results"])
    else:
        task_summaries = _load_existing_task_summaries(
            run_root,
            skip_tasks=set(external_tasks),
        )

    for task_name in external_tasks:
        task_output_dir = run_root / task_name
        external_args = argparse.Namespace(**vars(args))
        external_args.model = resolved["model"]
        external_args.model_name = resolved["model_name"]
        external_args.backend = resolved["backend"]
        external_args.output_dir = str(task_output_dir)
        external_args.dp_size = resolved["dp_size"]
        external_args.tp_size = resolved["tp_size"]
        if args.num_gpus is not None and args.dp_size is None:
            # Preserve the legacy BFCL-only override instead of replacing it with
            # resolve_run_arguments()'s generic dp_size default.
            external_args.dp_size = None
        external_args.overwrite = resolved["overwrite"]
        external_args.generate_only = resolved["generate_only"]
        external_args.eval_only = resolved["eval_only"]

        gen_overrides = resolved["gen_overrides"]
        external_args.max_new_tokens = gen_overrides["max_new_tokens"]
        external_args.temperature = gen_overrides["temperature"]
        external_args.top_p = gen_overrides["top_p"]
        external_args.top_k = gen_overrides["top_k"]
        external_args.seed = gen_overrides["seed"]

        backend_kwargs = resolved["backend_kwargs"]
        external_args.gpu_memory_utilization = backend_kwargs.get(
            "gpu_memory_utilization",
            external_args.gpu_memory_utilization,
        )
        external_args.context_length = backend_kwargs.get("context_length")
        external_args.max_model_len = backend_kwargs.get("max_model_len")
        external_args.backend_kwargs = backend_kwargs
        spec, run = _build_external_spec(
            external_args,
            task_name=task_name,
            output_dir=task_output_dir,
        )
        _info(
            f"external_task={task_name} model={external_args.model} "
            f"backend={spec.backend} dp_size={spec.dp_size} tp_size={spec.tp_size} "
            f"router={spec.use_sglang_router} output_dir={spec.output_dir}"
        )
        result = run(spec)
        task_summaries[task_name] = _external_summary(
            task_name,
            task_output_dir,
            result,
        )

    if not external_tasks and native_result is not None:
        return native_result

    return _write_combined_run_summary(
        run_root=run_root,
        run_id=run_id,
        selected_tasks=native_tasks + external_tasks,
        model=str(resolved["model"]),
        model_name=effective_model_name,
        backend=str(resolved["backend"]),
        phase=(
            "generate_only"
            if resolved["generate_only"]
            else "eval_only"
            if resolved["eval_only"]
            else "generate_and_eval"
        ),
        task_summaries=task_summaries,
    )


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
        "--config", type=str, default=None, help="YAML config file path."
    )
    parser.add_argument(
        "--tasks", type=str, default=None, help="Task names: all or comma-separated."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Actual Hugging Face model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional logical/output name; does not affect model loading.",
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
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--generate-only",
        action="store_true",
        default=None,
        help=(
            "Generate and save predictions without running metrics or LLM judges. "
            "Resume with --eval-only using the same model/output/run-id."
        ),
    )
    phase_group.add_argument(
        "--eval-only",
        action="store_true",
        default=None,
        help=(
            "Evaluate a complete existing predictions.jsonl without loading the "
            "candidate inference backend."
        ),
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
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pass enable_thinking=true/false to the tokenizer chat template. "
            "Omit both forms to preserve the checkpoint's native default."
        ),
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
        "--judge-model",
        type=str,
        default=None,
        help="Override the benchmark's aligned default LLM judge model.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("api", "local"),
        default=None,
        help=(
            "Judge transport: OpenAI-compatible API or an internally managed "
            "offline SGLang engine (default: api)."
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible judge API base URL (or AETHEREVAL_JUDGE_BASE_URL).",
    )
    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default=None,
        help="Environment variable containing the judge API key.",
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=None,
        help="Concurrent LLM-judge requests (default: 64).",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="Per-request LLM-judge timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=None,
        help="Transport retry count for LLM-judge requests (default: 5).",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=None,
        help="Override benchmark-specific judge repetition count.",
    )
    parser.add_argument(
        "--judge-dp-size",
        type=int,
        default=None,
        help=(
            "Offline judge data-parallel size. If judge DP/TP are both omitted, "
            "the judge uses TP=runtime DP*TP."
        ),
    )
    parser.add_argument(
        "--judge-tp-size",
        type=int,
        default=None,
        help="Offline judge tensor-parallel size.",
    )
    parser.add_argument(
        "--judge-local-max-tokens",
        type=int,
        default=None,
        help="Offline judge max-token fallback when a benchmark does not specify one.",
    )
    parser.add_argument(
        "--judge-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the offline judge model's thinking chat-template mode.",
    )
    parser.add_argument(
        "--judge-sglang-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra offline judge SGLang Engine kwarg; repeat as needed.",
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
        help="BFCL categories/collections, comma-separated.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help=(
            "Legacy BFCL total GPU count; for SGLang it maps to DP replicas when "
            "--dp-size/--tp-size are omitted."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="BFCL local request concurrency; defaults to max(16, 16 * dp_size), capped at 100.",
    )
    parser.add_argument(
        "--bfcl-use-sglang-router",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use SGLang Model Gateway for BFCL when dp_size > 1 (default: enabled).",
    )
    parser.add_argument(
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
        help="SGLang Model Gateway routing policy for BFCL (default: cache_aware).",
    )
    parser.add_argument(
        "--bfcl-context-length",
        type=int,
        default=None,
        help=(
            "BFCL-only SGLang context length; overrides --context-length after "
            "native tasks have finished."
        ),
    )
    parser.add_argument(
        "--bfcl-sglang-arg",
        action="append",
        default=None,
        help=(
            "Extra BFCL-only SGLang server argument (repeatable), format: key=value. "
            "Overrides the matching global --sglang-arg."
        ),
    )
    parser.add_argument(
        "--bfcl-verbose",
        action="store_true",
        help="Show verbose BFCL multi-turn step logs.",
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
        help="External tasks: reuse existing raw generations.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="External tasks: generate only and skip scoring.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_tasks:
        for task_name in sorted(set(list_tasks()) | set(EXTERNAL_TASKS)):
            print(task_name)
        return
    if args.list_task_defaults:
        payload = list_task_default_gens()
        payload.update(
            {
                task_name: resolve_task_default_gen(task_name, {})
                for task_name in EXTERNAL_TASKS
            }
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    cfg = load_yaml_config(args.config)
    resolved = resolve_run_arguments(args, cfg)

    if not resolved["model"]:
        parser.error("--model is required unless --list-tasks is set.")

    _info(f"config={args.config if args.config else '(none)'}")
    _info(
        f"model={resolved['model']} "
        f"model_name={model_output_name(str(resolved['model']), resolved['model_name'])} "
        f"backend={resolved['backend']} tasks={resolved['tasks']} "
        f"dp_size={resolved['dp_size']} tp_size={resolved['tp_size']} "
        f"overwrite={resolved['overwrite']}"
    )
    phase = (
        "generate_only"
        if resolved["generate_only"]
        else "eval_only"
        if resolved["eval_only"]
        else "generate_and_eval"
    )
    _info(f"phase={phase}")
    _info(
        f"output_dir={resolved['output_dir']} "
        f"run_id={resolved['run_id'] if resolved['run_id'] else '(auto:model_name)'}"
    )
    explicit_gen_overrides = {
        k: v for k, v in resolved["gen_overrides"].items() if v is not None
    }
    if explicit_gen_overrides:
        _info(f"generation_overrides={explicit_gen_overrides}")
    if resolved["backend_kwargs"]:
        _info(f"backend_model_kwargs={resolved['backend_kwargs']}")

    try:
        result = run_selected_tasks(args, resolved)
    except ValueError as exc:
        parser.error(str(exc))

    if "inspect" in result:
        inspected = result["inspect"]
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

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
