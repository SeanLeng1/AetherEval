import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from benchmarks.bfcl.cli import add_bfcl_arguments, build_bfcl_spec
from benchmarks.bfcl.external import run as run_bfcl

from .config import load_yaml_config, resolve_run_arguments
from .core.io import ensure_dir, model_output_name, run_output_dir
from .core.run_summary import build_run_summary, load_task_summaries, phase_name
from .core.runner import inspect_prompts, run_evaluation
from .core.task_defaults import resolve_task_default_gen
from .core.task_register import list_task_default_gens, list_tasks, parse_task_names

EXTERNAL_TASKS = ("bfcl",)


def _info(message: str) -> None:
    print(f"[aethereval] {message}")


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


def _split_native_external_tasks(tasks_arg: str) -> tuple[list[str], list[str]]:
    native_available = set(list_tasks())
    selected = parse_task_names(tasks_arg, native_available | set(EXTERNAL_TASKS))
    native_tasks = [name for name in selected if name in native_available]
    external_tasks = [name for name in selected if name in EXTERNAL_TASKS]
    return native_tasks, external_tasks


def _external_summary(
    task_name: str, task_output_dir: Path, result: Any
) -> dict[str, Any]:
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


def run_selected_tasks(
    args: argparse.Namespace,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    if resolved["inspect"] and (resolved["generate_only"] or resolved["eval_only"]):
        raise ValueError("--inspect cannot be combined with a phase-only mode")
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
            backend_kwargs=resolved["backend_kwargs"],
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

        if not resolved["generate_only"] and not resolved["eval_only"]:
            _info(
                "two-phase execution: generating all native tasks first, then "
                "restarting in eval-only mode"
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
        task_summaries = load_task_summaries(
            run_root,
            skip_tasks=set(external_tasks),
        )

    for task_name in external_tasks:
        task_output_dir = run_root / task_name
        if task_name != "bfcl":
            raise ValueError(f"Unknown external task: {task_name}")
        spec = build_bfcl_spec(args, resolved, task_output_dir)
        _info(
            f"external_task={task_name} model={spec.model} "
            f"backend={spec.backend} dp_size={spec.dp_size} tp_size={spec.tp_size} "
            f"num_runs={spec.num_runs} output_dir={spec.output_dir}"
        )
        result = run_bfcl(spec)
        task_summaries[task_name] = _external_summary(
            task_name,
            task_output_dir,
            result,
        )

    if not external_tasks and native_result is not None:
        return native_result

    return build_run_summary(
        run_root=run_root,
        run_id=run_id,
        selected_tasks=native_tasks + external_tasks,
        model=str(resolved["model"]),
        model_name=effective_model_name,
        backend=str(resolved["backend"]),
        phase=phase_name(
            generate_only=resolved["generate_only"],
            eval_only=resolved["eval_only"],
        ),
        task_summaries=task_summaries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AetherEval: lightweight generative-only LLM eval framework."
    )
    run_group = parser.add_argument_group("run")
    run_group.add_argument(
        "--list-tasks", action="store_true", help="List discovered tasks and exit."
    )
    run_group.add_argument(
        "--list-task-defaults",
        action="store_true",
        help="Print effective DEFAULT_GEN for all tasks and exit.",
    )
    run_group.add_argument(
        "--config", type=str, default=None, help="YAML config file path."
    )
    run_group.add_argument(
        "--tasks", type=str, default=None, help="Task names: all or comma-separated."
    )
    run_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Actual Hugging Face model ID or local checkpoint path.",
    )
    run_group.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional logical/output name; does not affect model loading.",
    )
    run_group.add_argument(
        "--inspect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print first 5 prompts after chat-template rendering and exit (no inference).",
    )
    run_group.add_argument(
        "--output-dir", type=str, default=None, help="Output root directory."
    )
    run_group.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. Default: <model_suffix_lower>.",
    )
    run_group.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing predictions.jsonl for the same run_id.",
    )
    phase_group = run_group.add_mutually_exclusive_group()
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

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--backend",
        type=str,
        choices=("vllm", "sglang"),
        default=None,
        help="Inference backend.",
    )
    runtime_group.add_argument(
        "--dp-size", type=int, default=None, help="Data parallel worker count."
    )
    runtime_group.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size per worker.",
    )

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument(
        "--n", type=int, default=None, help="Override number of generations per sample."
    )
    generation_group.add_argument(
        "--max-new-tokens", type=int, default=None, help="Override max new tokens."
    )
    generation_group.add_argument(
        "--temperature", type=float, default=None, help="Override temperature."
    )
    generation_group.add_argument(
        "--top-p", type=float, default=None, help="Override top-p."
    )
    generation_group.add_argument(
        "--top-k", type=int, default=None, help="Override top-k (default: -1)."
    )
    generation_group.add_argument(
        "--min-p", type=float, default=None, help="Override min-p."
    )
    generation_group.add_argument(
        "--seed", type=int, default=None, help="Override sampling seed."
    )
    generation_group.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pass enable_thinking=true/false to the tokenizer chat template. "
            "Omit both forms to preserve the checkpoint's native default."
        ),
    )

    metrics_group = parser.add_argument_group("metrics")
    metrics_group.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=None,
        help="Bootstrap resample count forwarded to benchmark metrics.",
    )
    metrics_group.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Bootstrap RNG seed forwarded to benchmark metrics.",
    )
    metrics_group.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=None,
        help="Bootstrap confidence level in [0,1], forwarded to benchmark metrics.",
    )
    metrics_group.add_argument(
        "--rm-model-path",
        type=str,
        default=None,
        help="Reward-model path forwarded to RM-based benchmark metrics.",
    )
    metrics_group.add_argument(
        "--cm-model-path",
        type=str,
        default=None,
        help="Cost/safety-model path forwarded to RM-based benchmark metrics.",
    )
    metrics_group.add_argument(
        "--rm-max-length",
        type=int,
        default=None,
        help="Maximum token length for RM-based benchmark metrics.",
    )
    metrics_group.add_argument(
        "--rm-dtype",
        type=str,
        default=None,
        help="Torch dtype for RM-based benchmark metrics, e.g. bfloat16.",
    )
    metrics_group.add_argument(
        "--rm-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward trust_remote_code to RM tokenizers/SGLang servers.",
    )
    metrics_group.add_argument(
        "--rm-sglang-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra RM/CM SGLang server argument; repeat for multiple values.",
    )
    judge_group = parser.add_argument_group("LLM judge")
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override the benchmark's aligned default LLM judge model.",
    )
    judge_group.add_argument(
        "--judge-backend",
        choices=("api", "local"),
        default=None,
        help=(
            "Judge transport: LiteLLM API or an internally managed "
            "local SGLang service (default: api)."
        ),
    )
    judge_group.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help=(
            "Optional OpenAI-compatible judge endpoint for LiteLLM "
            "(or AETHEREVAL_JUDGE_BASE_URL). Omit for native provider routing."
        ),
    )
    judge_group.add_argument(
        "--judge-api-key-env",
        type=str,
        default=None,
        help="Environment variable containing the judge API key.",
    )
    judge_group.add_argument(
        "--judge-workers",
        type=int,
        default=None,
        help="Concurrent LLM-judge requests (default: 64).",
    )
    judge_group.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="Per-request LLM-judge timeout in seconds (default: 300).",
    )
    judge_group.add_argument(
        "--judge-max-retries",
        type=int,
        default=None,
        help="Transport retry count for LLM-judge requests (default: 5).",
    )
    judge_group.add_argument(
        "--judge-repeats",
        type=int,
        default=None,
        help="Override benchmark-specific judge repetition count.",
    )
    judge_group.add_argument(
        "--judge-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override judge max new tokens for every selected task. Omit to use "
            "each task's aligned default."
        ),
    )
    judge_group.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help=(
            "Override judge temperature for every selected task. Omit to use "
            "each task's aligned default."
        ),
    )
    judge_group.add_argument(
        "--judge-top-p",
        type=float,
        default=None,
        help=(
            "Override judge top-p for every selected task. Omit to use each "
            "task's aligned default."
        ),
    )
    judge_group.add_argument(
        "--judge-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pass enable_thinking=true/false to the judge chat template. Omit "
            "both forms to preserve each task/backend default."
        ),
    )
    judge_group.add_argument(
        "--judge-dp-size",
        type=int,
        default=None,
        help=(
            "Offline judge data-parallel size. If judge DP/TP are both omitted, "
            "the judge uses TP=runtime DP*TP."
        ),
    )
    judge_group.add_argument(
        "--judge-tp-size",
        type=int,
        default=None,
        help="Offline judge tensor-parallel size.",
    )
    judge_group.add_argument(
        "--judge-sglang-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra offline judge SGLang Engine kwarg; repeat as needed.",
    )

    backend_group = parser.add_argument_group("backend")
    backend_group.add_argument(
        "--gpu-memory-utilization", type=float, default=None, help="vLLM model kwarg."
    )
    backend_group.add_argument(
        "--max-model-len", type=int, default=None, help="vLLM model kwarg."
    )
    backend_group.add_argument(
        "--mem-fraction-static", type=float, default=None, help="SGLang Engine kwarg."
    )
    backend_group.add_argument(
        "--context-length", type=int, default=None, help="SGLang Engine kwarg."
    )
    backend_group.add_argument(
        "--dtype", type=str, default=None, help="Backend model kwarg."
    )
    backend_group.add_argument(
        "--vllm-arg",
        action="append",
        default=None,
        help="Extra vLLM model kwargs (repeatable), format: key=value",
    )
    backend_group.add_argument(
        "--sglang-arg",
        action="append",
        default=None,
        help="Extra SGLang Engine kwargs (repeatable), format: key=value",
    )

    add_bfcl_arguments(parser)
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
    phase = phase_name(
        generate_only=resolved["generate_only"],
        eval_only=resolved["eval_only"],
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
