"""External-benchmark API for BFCL V3 in AetherEval."""

import builtins
import contextlib
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from aethereval.backends.sglang.service import SGLangService
from aethereval.core.io import model_output_name
from aethereval.core.task_defaults import (
    resolve_task_default_gen,
    resolve_task_num_repeats,
)

from .errors import is_context_length_error
from .register import register_rlla_model
from .scoring import install_scalar_quotation_tolerance

# ToolRL's public format reward selects one exact shape from the reference answer and
# checks both the regex and tag counts. BFCL does not ship ToolRL-formatted references,
# so the expected shape is derived from the BFCL subset and multi-turn ground truth.
_FMT_SPECS = {
    "tool_call": (
        r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$",
        ("<think>", "</think>", "<tool_call>", "</tool_call>"),
    ),
    "response": (
        r"^<think>.*?</think>\n<response>.*?</response>$",
        ("<think>", "</think>", "<response>", "</response>"),
    ),
}
_NO_CALL_CATEGORIES = {"irrelevance", "live_irrelevance"}
_NOISY_BFCL_MESSAGES = {
    "Empty response from the model. Proceed to next turn.",
    "Failed to decode the model response. Proceed to next turn.",
}
_DEFAULT_GEN = resolve_task_default_gen("bfcl", {})
_DEFAULT_NUM_REPEATS = resolve_task_num_repeats("bfcl")
DEFAULT_CATEGORIES = tuple(
    _DEFAULT_GEN.get("categories", ("live", "non_live", "multi_turn"))
)
_COMPARISON_SECTIONS = (
    ("live", "live_acc", "live_format"),
    ("non_live", "non_live_acc", "non_live_format"),
    ("multi_turn", "multi_turn_acc", "multi_turn_format"),
)


@dataclass
class ExternalRunSpec:
    model: str  # Hugging Face id or local checkpoint path
    output_dir: Path  # AetherEval run dir; result/ + score/ go under it
    model_name: str | None = None  # Logical/output label; never used for loading
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    backend: str = "sglang"  # tmux0 container ships sglang
    dp_size: int = 1
    tp_size: int = 1
    router_policy: str = "cache_aware"
    num_threads: int = 16
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    sglang_server_args: dict[str, Any] = field(default_factory=dict)
    temperature: float = float(
        _DEFAULT_GEN.get("temperature", 0.001)
    )  # near-greedy, BFCL tool-calling default
    max_tokens: int = int(_DEFAULT_GEN.get("max_new_tokens", 4096))
    max_context_length: int | None = None
    top_p: float = float(_DEFAULT_GEN.get("top_p", 1.0))
    top_k: int = int(_DEFAULT_GEN.get("top_k", -1))
    repetition_penalty: float = 1.0
    seed: int | None = None
    verbose: bool = False
    allow_overwrite: bool = True
    run_generation: bool = True
    run_evaluation: bool = True
    num_repeats: int = _DEFAULT_NUM_REPEATS

    @property
    def num_gpus(self) -> int:
        return self.dp_size * self.tp_size


@dataclass
class ExternalResult:
    metrics: dict[str, float]
    primary_metric: str
    primary_score: float
    result_dir: Path
    score_dir: Path


def _gen_args(
    spec: ExternalRunSpec,
    result_dir: Path,
    *,
    skip_server_setup: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=[spec.model],
        test_category=list(spec.categories),
        temperature=spec.temperature,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=spec.num_gpus,
        num_threads=spec.num_threads,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        backend=spec.backend,
        skip_server_setup=skip_server_setup,
        local_model_path=spec.model if Path(spec.model).is_dir() else None,
        result_dir=result_dir,  # absolute -> PROJECT_ROOT / abs == abs
        allow_overwrite=spec.allow_overwrite,
        run_ids=False,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )


@contextlib.contextmanager
def _temporary_env(values: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _handler_env(spec: ExternalRunSpec) -> dict[str, str | None]:
    return {
        "RLLA_BFCL_MAX_TOKENS": str(spec.max_tokens),
        "RLLA_BFCL_MAX_CONTEXT_LENGTH": (
            str(spec.max_context_length)
            if spec.max_context_length is not None
            else None
        ),
        "RLLA_BFCL_TOP_P": str(spec.top_p),
        "RLLA_BFCL_TOP_K": str(spec.top_k),
        "RLLA_BFCL_REPETITION_PENALTY": str(spec.repetition_penalty),
        "RLLA_BFCL_SEED": str(spec.seed) if spec.seed is not None else None,
    }


@contextlib.contextmanager
def _cap_bfcl_thread_pool(max_workers: int):
    """Cap BFCL V3's hard-coded 100-request local inference pool."""
    if max_workers <= 0:
        raise ValueError("BFCL num_threads must be positive.")

    from bfcl_eval.model_handler.local_inference import base_oss_handler

    original = base_oss_handler.ThreadPoolExecutor

    def capped_thread_pool_executor(*args, **kwargs):
        requested = kwargs.get("max_workers")
        if requested is None and args:
            requested = args[0]
        capped = max_workers if requested is None else min(int(requested), max_workers)
        if args:
            args = (capped, *args[1:])
        else:
            kwargs["max_workers"] = capped
        return original(*args, **kwargs)

    base_oss_handler.ThreadPoolExecutor = capped_thread_pool_executor
    try:
        yield
    finally:
        base_oss_handler.ThreadPoolExecutor = original


def _remove_command_options(command: list[str], option_names: set[str]) -> list[str]:
    cleaned: list[str] = []
    skip_value = False
    for item in command:
        if skip_value:
            skip_value = False
            continue
        if item in option_names:
            skip_value = True
            continue
        if any(item.startswith(f"{name}=") for name in option_names):
            continue
        cleaned.append(item)
    return cleaned


def _server_command_for_spec(command, spec: ExternalRunSpec):
    if not isinstance(command, (list, tuple)):
        return command

    patched = list(command)
    if patched[:2] == ["vllm", "serve"] and spec.max_context_length is not None:
        patched = _remove_command_options(patched, {"--max-model-len"})
        patched.extend(["--max-model-len", str(spec.max_context_length)])

    return type(command)(patched) if isinstance(command, tuple) else patched


@contextlib.contextmanager
def _patch_bfcl_server_command(spec: ExternalRunSpec):
    from bfcl_eval.model_handler.local_inference import base_oss_handler

    original = base_oss_handler.subprocess.Popen

    def patched_popen(*args, **kwargs):
        if args:
            args = (
                _server_command_for_spec(args[0], spec),
                *args[1:],
            )
        elif "args" in kwargs:
            kwargs["args"] = _server_command_for_spec(kwargs["args"], spec)
        env = dict(kwargs.get("env") or os.environ)
        warning_filters = env.get("PYTHONWARNINGS", "")
        quiet_filters = "ignore::SyntaxWarning,ignore::FutureWarning"
        env["PYTHONWARNINGS"] = ",".join(
            value for value in (warning_filters, quiet_filters) if value
        )
        env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
        env["TQDM_DISABLE"] = "1"
        kwargs["env"] = env
        return original(*args, **kwargs)

    base_oss_handler.subprocess.Popen = patched_popen
    try:
        yield
    finally:
        base_oss_handler.subprocess.Popen = original


def _is_noisy_bfcl_print(args: tuple[object, ...]) -> bool:
    text = " ".join(str(arg) for arg in args).strip()
    if text in _NOISY_BFCL_MESSAGES:
        return True
    if text.startswith("ID: ") and ", Turn: " in text and ", Step: " in text:
        return True
    return len(text) >= 80 and set(text) == {"-"}


@contextlib.contextmanager
def _filter_bfcl_prints(enabled: bool):
    if not enabled:
        yield
        return

    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if _is_noisy_bfcl_print(args):
            return
        original_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


def _run_generation(
    spec: ExternalRunSpec,
    result_dir: Path,
    generation_main,
) -> None:
    _run_generations(spec, [(0, result_dir)], generation_main)


def _run_generations(
    spec: ExternalRunSpec,
    runs: list[tuple[int, Path]],
    generation_main,
) -> None:
    if spec.num_threads <= 0:
        raise ValueError("BFCL num_threads must be positive.")
    if spec.backend == "sglang":
        service = SGLangService(
            model=spec.model,
            dp_size=spec.dp_size,
            tensor_parallel_size=spec.tp_size,
            model_kwargs=spec.sglang_server_args,
            router_policy=spec.router_policy,
        )
        try:
            endpoint = urlsplit(service.base_url)
            if endpoint.hostname is None or endpoint.port is None:
                raise RuntimeError(
                    f"Invalid AetherEval SGLang endpoint: {service.base_url}"
                )
            with (
                _temporary_env(
                    {
                        "VLLM_ENDPOINT": endpoint.hostname,
                        "VLLM_PORT": str(endpoint.port),
                        "LOCAL_SERVER_ENDPOINT": endpoint.hostname,
                        "LOCAL_SERVER_PORT": str(endpoint.port),
                        "RLLA_BFCL_GENERATE_URL": f"{service.base_url}/generate",
                    }
                ),
                _filter_bfcl_prints(not spec.verbose),
                _cap_bfcl_thread_pool(spec.num_threads),
            ):
                for run_index, result_dir in runs:
                    run_spec = replace(spec, seed=_repeat_seed(spec, run_index))
                    with _temporary_env(_handler_env(run_spec)):
                        generation_main(
                            _gen_args(run_spec, result_dir, skip_server_setup=True)
                        )
        finally:
            service.close()
        return

    for run_index, result_dir in runs:
        run_spec = replace(spec, seed=_repeat_seed(spec, run_index))
        with (
            _temporary_env(_handler_env(run_spec)),
            _filter_bfcl_prints(not spec.verbose),
            _cap_bfcl_thread_pool(run_spec.num_threads),
            _patch_bfcl_server_command(run_spec),
        ):
            generation_main(_gen_args(run_spec, result_dir))


def _repeat_seed(spec: ExternalRunSpec, repeat_index: int) -> int:
    return (spec.seed if spec.seed is not None else 0) + repeat_index


def _evaluation_run_paths(out: Path, num_repeats: int) -> list[tuple[Path, Path]]:
    if num_repeats <= 0:
        raise ValueError("BFCL num_repeats must be positive.")
    result_root = out / "result"
    score_root = out / "score"
    if num_repeats == 1:
        return [(result_root, score_root)]
    return [
        (result_root / f"run_{index + 1:02d}", score_root / f"run_{index + 1:02d}")
        for index in range(num_repeats)
    ]


def _next_corrupt_tail_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.corrupt-tail")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.corrupt-tail.{index}")
        index += 1
    return candidate


def _validate_or_repair_result_jsonl(path: Path, *, repair_tail: bool) -> None:
    """Validate one BFCL result file and recover an interrupted final write."""

    with path.open("rb") as source:
        line_index = 0
        while True:
            byte_offset = source.tell()
            raw_line = source.readline()
            if not raw_line:
                return
            line_index += 1
            if not raw_line.strip():
                continue
            try:
                json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                has_later_record = any(line.strip() for line in source)
                if repair_tail and not has_later_record:
                    backup_path = _next_corrupt_tail_path(path)
                    source.seek(byte_offset)
                    with backup_path.open("wb") as backup:
                        shutil.copyfileobj(source, backup)
                    with path.open("r+b") as result_file:
                        result_file.truncate(byte_offset)
                    print(
                        "[aethereval] BFCL resume repaired an interrupted final JSONL "
                        f"record: file={path} line={line_index} "
                        f"backup={backup_path}"
                    )
                    return
                action = (
                    "Rerun generation so AetherEval can repair the interrupted tail."
                    if not repair_tail and not has_later_record
                    else "Restore or remove this result file before resuming."
                )
                raise RuntimeError(
                    "Malformed BFCL result JSONL: "
                    f"file={path}, line={line_index}: {exc}. {action}"
                ) from exc


def _prepare_existing_results(
    result_dirs: list[Path],
    model: str,
    *,
    repair_tail: bool,
) -> None:
    model_dir_name = model.replace("/", "_")
    for result_dir in result_dirs:
        model_dir = result_dir / model_dir_name
        if not model_dir.exists():
            continue
        for result_file in sorted(model_dir.rglob("*_result.json")):
            _validate_or_repair_result_jsonl(result_file, repair_tail=repair_tail)


def run(spec: ExternalRunSpec) -> ExternalResult:
    out = Path(spec.output_dir).resolve()
    run_paths = _evaluation_run_paths(out, spec.num_repeats)
    for result_dir, score_dir in run_paths:
        result_dir.mkdir(parents=True, exist_ok=True)
        score_dir.mkdir(parents=True, exist_ok=True)

    if not spec.allow_overwrite:
        _prepare_existing_results(
            [result_dir for result_dir, _ in run_paths],
            spec.model,
            repair_tail=spec.run_generation,
        )

    if spec.run_generation or spec.run_evaluation:
        register_rlla_model(spec.model, project_root=str(out))

    if spec.run_generation:
        from bfcl_eval._llm_response_generation import main as generation_main

        _run_generations(
            spec,
            [(index, paths[0]) for index, paths in enumerate(run_paths)],
            generation_main,
        )

    repeat_metrics: list[dict[str, float]] = []
    repeat_summaries: list[dict[str, Any]] = []
    prediction_stats: dict[str, int | str] = {
        "predictions_path": str(out / "predictions.jsonl"),
        "prediction_records": 0,
        "prediction_scored_records": 0,
    }
    for run_index, (result_dir, score_dir) in enumerate(run_paths):
        if spec.run_generation or spec.run_evaluation:
            _raise_on_inference_errors(result_dir, spec.model)

        if spec.run_evaluation:
            from bfcl_eval.eval_checker import eval_runner

            install_scalar_quotation_tolerance()
            eval_runner.main(
                [spec.model], list(spec.categories), str(result_dir), str(score_dir)
            )

        metrics = parse_scores(score_dir) if spec.run_evaluation else {}
        add_comparison_metrics(metrics, compute_format_rates(result_dir, spec.model))
        if spec.run_evaluation and not metrics:
            raise RuntimeError(
                f"BFCL evaluation repeat {run_index + 1} produced no metrics."
            )
        repeat_metrics.append(metrics)
        stats = write_predictions_jsonl(
            out=out,
            result_dir=result_dir,
            score_dir=score_dir,
            model=spec.model,
            gen_idx=run_index,
            append=run_index > 0,
        )
        prediction_stats["prediction_records"] = int(
            prediction_stats["prediction_records"]
        ) + int(stats["prediction_records"])
        prediction_stats["prediction_scored_records"] = int(
            prediction_stats["prediction_scored_records"]
        ) + int(stats["prediction_scored_records"])
        repeat_summaries.append(
            {
                "repeat": run_index + 1,
                "seed": _repeat_seed(spec, run_index),
                "result_dir": str(result_dir),
                "score_dir": str(score_dir),
                "metrics": metrics,
            }
        )

    metrics = average_repeat_metrics(repeat_metrics)
    primary_metric = "overall_acc"
    primary_score = float(metrics.get(primary_metric, 0.0))
    _write_summary(
        out,
        spec,
        metrics,
        primary_metric,
        primary_score,
        prediction_stats,
        repeat_summaries,
    )
    return ExternalResult(
        metrics,
        primary_metric,
        primary_score,
        out / "result",
        out / "score",
    )


def average_repeat_metrics(repeats: list[dict[str, float]]) -> dict[str, float]:
    if not repeats:
        return {}
    common_keys = set(repeats[0]).intersection(
        *(set(metrics) for metrics in repeats[1:])
    )
    averaged = {
        key: round(sum(metrics[key] for metrics in repeats) / len(repeats), 2)
        for key in common_keys
    }
    _add_overall_metrics(averaged)
    return averaged


def _read_overall_csv(score_dir: Path) -> dict[str, float] | None:
    """Parse BFCL V3's official leaderboard CSV into the four report scores."""
    csv_path = score_dir / "data_overall.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]  # single registered model

    def pct(*columns: str) -> float | None:
        value = None
        for column in columns:
            value = row.get(column)
            if value is not None:
                break
        if value is None:
            return None
        try:
            return float(str(value).replace("%", "").strip())
        except ValueError:
            return None  # "N/A" (category not run)

    report_mapping = {
        "overall_acc": ("Overall Acc",),
        "non_live_acc": ("Non-Live AST Acc",),
        "live_acc": ("Live Acc",),
        "multi_turn_acc": ("Multi Turn Acc",),
    }
    out: dict[str, float] = {}
    for metric, columns in report_mapping.items():
        value = pct(*columns)
        if value is not None:
            out[metric] = value

    return out or None


def parse_scores(score_dir: Path) -> dict[str, float]:
    """Read only BFCL V3's official aggregate report; never synthesize scores."""
    metrics = _read_overall_csv(score_dir)
    if not metrics:
        raise RuntimeError(
            "BFCL evaluation did not produce a parseable official report at "
            f"{score_dir / 'data_overall.csv'}."
        )
    _add_overall_metrics(metrics)
    return metrics


def _is_allowed_zero_score_error(error: str) -> bool:
    return is_context_length_error(error)


def _raise_on_inference_errors(result_dir: Path, model: str) -> None:
    model_dir = result_dir / model.replace("/", "_")
    result_files = list(model_dir.rglob("*_result.json")) if model_dir.exists() else []

    count = 0
    examples: list[str] = []
    for jf in result_files:
        with open(jf) as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                result = record["result"]
                errors = [
                    str(value)
                    for value in _iter_result_values(result)
                    if str(value).startswith("Error during inference:")
                    and not _is_allowed_zero_score_error(str(value))
                ]
                if not errors:
                    continue

                count += 1
                if len(examples) < 5:
                    examples.append(
                        f"{record['id']} ({jf.name}:{line_no}): {errors[0]}"
                    )

    if count:
        raise RuntimeError(
            "BFCL generation produced inference errors instead of model outputs. "
            f"count={count}; examples={'; '.join(examples)}"
        )


def compute_format_rates(result_dir: Path, model: str) -> dict[str, float]:
    """Reference-aware ToolRL format percentages for BFCL comparison sections."""

    model_dir = result_dir / model.replace("/", "_")
    counts = {section: {"total": 0, "ok": 0} for section, _, _ in _COMPARISON_SECTIONS}
    for jf in model_dir.rglob("*_result.json"):
        category = _result_file_category(jf)
        section = _comparison_section(category)
        if section not in counts:
            continue
        multi_turn_ground_truth = (
            _load_ground_truth_by_id(category)
            if category.startswith("multi_turn_")
            else None
        )
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for response, expected in _iter_expected_format_outputs(
                    category, record, multi_turn_ground_truth
                ):
                    counts[section]["total"] += 1
                    if _has_expected_toolrl_format(response, expected):
                        counts[section]["ok"] += 1
    return {
        section: 100.0 * count["ok"] / count["total"]
        for section, count in counts.items()
        if count["total"]
    }


def _normalize_toolrl_response(value: Any) -> str:
    text = str(value).strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        while text.endswith(token):
            text = text[: -len(token)].rstrip()
    return text


def _has_expected_toolrl_format(value: Any, expected: str) -> bool:
    text = _normalize_toolrl_response(value)
    pattern, tags = _FMT_SPECS[expected]
    return bool(re.search(pattern, text, re.DOTALL)) and all(
        text.count(tag) == 1 for tag in tags
    )


def _load_ground_truth_by_id(category: str) -> dict[str, Any]:
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH
    from bfcl_eval.utils import find_file_with_suffix, load_file

    return {
        str(entry["id"]): entry["ground_truth"]
        for entry in load_file(find_file_with_suffix(POSSIBLE_ANSWER_PATH, category))
    }


def _iter_expected_format_outputs(
    category: str,
    record: dict[str, Any],
    multi_turn_ground_truth: dict[str, Any] | None,
):
    result = record.get("result", "")
    if not category.startswith("multi_turn_"):
        expected = "response" if category in _NO_CALL_CATEGORIES else "tool_call"
        for response in _iter_result_values(result):
            yield response, expected
        return

    sample_id = str(record["id"])
    if multi_turn_ground_truth is None or sample_id not in multi_turn_ground_truth:
        raise RuntimeError(
            "BFCL format scoring could not find multi-turn ground truth for "
            f"{sample_id!r} in category {category!r}."
        )
    ground_truth = multi_turn_ground_truth[sample_id]
    turn_results = result if isinstance(result, list) else [result]
    for turn_index, turn_result in enumerate(turn_results):
        outputs = list(_iter_result_values(turn_result))
        needs_tool = turn_index < len(ground_truth) and bool(ground_truth[turn_index])
        if not needs_tool:
            for response in outputs:
                yield response, "response"
            continue

        # A tool-requiring turn must first call a tool and, after execution,
        # terminate with a response. A single premature response is therefore
        # checked as a tool call; a forced quit after only calls fails its final
        # completion because the terminal response is missing.
        for output_index, response in enumerate(outputs):
            expected = (
                "response"
                if len(outputs) > 1 and output_index == len(outputs) - 1
                else "tool_call"
            )
            yield response, expected


def add_comparison_metrics(
    metrics: dict[str, float],
    format_rates: dict[str, float],
) -> None:
    """Add section format rates and V3-style macro overall scores."""
    for section, _, format_key in _COMPARISON_SECTIONS:
        if section in format_rates:
            metrics[format_key] = round(format_rates[section], 2)

    _add_overall_metrics(metrics)


def _add_overall_metrics(metrics: dict[str, float]) -> None:
    accuracy_keys = [item[1] for item in _COMPARISON_SECTIONS]
    if "overall_acc" not in metrics and all(key in metrics for key in accuracy_keys):
        metrics["overall_acc"] = round(
            sum(metrics[key] for key in accuracy_keys) / len(accuracy_keys),
            2,
        )

    format_keys = [item[2] for item in _COMPARISON_SECTIONS]
    if all(key in metrics for key in format_keys):
        metrics["overall_format"] = round(
            sum(metrics[key] for key in format_keys) / len(format_keys),
            2,
        )


def _as_generation_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _iter_result_values(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _iter_result_values(item)
        return
    yield value


def _result_error(value: Any) -> str | None:
    for item in _iter_result_values(value):
        if isinstance(item, str) and item.startswith("Error during inference:"):
            return item
    return None


def _result_file_category(path: Path) -> str:
    name = path.name
    for suffix in ("_result.json", "_score.json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = re.sub(r"^BFCL_v\d+_", "", name)
    return name


def _comparison_section(category: str) -> str | None:
    if category.startswith("multi_turn_"):
        return "multi_turn"
    if category.startswith("live_"):
        return "live"
    if category in {
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
    }:
        return "non_live"
    return None


def _bfcl_package_version() -> str:
    try:
        return version("bfcl-eval")
    except PackageNotFoundError:
        return "unknown"


def _score_file_for_result_file(
    *,
    result_file: Path,
    result_model_dir: Path,
    score_model_dir: Path,
) -> Path | None:
    relative = result_file.relative_to(result_model_dir)
    direct = score_model_dir / str(relative).replace("_result.json", "_score.json")
    if direct.exists():
        return direct

    stem = result_file.name.replace("_result.json", "_score.json")
    matches = list(score_model_dir.rglob(stem)) if score_model_dir.exists() else []
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple BFCL score files match {result_file.name}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _invalid_ids_from_score_file(score_file: Path | None) -> set[str] | None:
    if score_file is None:
        return None

    invalid_ids: set[str] = set()
    with score_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if line_no == 1 and "accuracy" in record:
                continue
            if "id" in record:
                invalid_ids.add(str(record["id"]))
    return invalid_ids


def _score_from_invalid_ids(
    sample_id: str,
    invalid_ids: set[str] | None,
) -> tuple[float | None, bool | None]:
    if invalid_ids is None:
        return None, None
    is_pass = sample_id not in invalid_ids
    return float(is_pass), is_pass


def _prompt_from_bfcl_record(record: dict[str, Any]) -> Any:
    input_log = record.get("inference_input_log")
    if isinstance(input_log, dict) and "formatted_prompt" in input_log:
        return input_log["formatted_prompt"]
    if "prompt" in record:
        return record["prompt"]
    return ""


def write_predictions_jsonl(
    *,
    out: Path,
    result_dir: Path,
    score_dir: Path,
    model: str,
    gen_idx: int = 0,
    append: bool = False,
) -> dict[str, int | str]:
    predictions_path = out / "predictions.jsonl"
    model_dir = result_dir / model.replace("/", "_")
    score_model_dir = score_dir / model.replace("/", "_")
    result_files = (
        sorted(model_dir.rglob("*_result.json")) if model_dir.exists() else []
    )

    total_records = 0
    scored_records = 0
    with predictions_path.open("a" if append else "w", encoding="utf-8") as f:
        for result_file in result_files:
            invalid_ids = _invalid_ids_from_score_file(
                _score_file_for_result_file(
                    result_file=result_file,
                    result_model_dir=model_dir,
                    score_model_dir=score_model_dir,
                )
            )
            category = _result_file_category(result_file)
            with result_file.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    raw_record = json.loads(line)
                    sample_id = str(raw_record["id"])
                    score, is_pass = _score_from_invalid_ids(sample_id, invalid_ids)
                    if score is not None:
                        scored_records += 1

                    raw_result = raw_record["result"]
                    row = {
                        "sample_id": sample_id,
                        "gen_idx": gen_idx,
                        "prompt": _prompt_from_bfcl_record(raw_record),
                        "generation": _as_generation_text(raw_result),
                        "score": score,
                        "is_pass": is_pass,
                        "parsed": raw_result,
                        "gold": None,
                        "error": _result_error(raw_result),
                        "meta": {
                            "benchmark": "bfcl",
                            "evaluation_repeat": gen_idx + 1,
                            "test_category": category,
                            "result_file": str(result_file.relative_to(result_dir)),
                            "score_available": score is not None,
                            "bfcl_record": {
                                key: value
                                for key, value in raw_record.items()
                                if key != "result"
                            },
                        },
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_records += 1

    return {
        "predictions_path": str(predictions_path),
        "prediction_records": total_records,
        "prediction_scored_records": scored_records,
    }


def _write_summary(
    out: Path,
    spec: ExternalRunSpec,
    metrics: dict[str, float],
    primary_metric: str,
    primary_score: float,
    prediction_stats: dict[str, int | str],
    repeat_summaries: list[dict[str, Any]],
) -> None:
    summary = {
        "benchmark": "bfcl",
        "benchmark_version": "v3",
        "bfcl_eval_version": _bfcl_package_version(),
        "external": True,
        "model": spec.model,
        "model_name": model_output_name(spec.model, spec.model_name),
        "backend": spec.backend,
        "categories": list(spec.categories),
        "num_gpus": spec.num_gpus,
        "dp_size": spec.dp_size,
        "tp_size": spec.tp_size,
        "router_policy": spec.router_policy,
        "num_threads": spec.num_threads,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "dtype": spec.dtype,
        "sglang_server_args": spec.sglang_server_args,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "max_context_length": spec.max_context_length,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "repetition_penalty": spec.repetition_penalty,
        "seed": spec.seed,
        "num_repeats": spec.num_repeats,
        "repeat_seeds": [
            _repeat_seed(spec, index) for index in range(spec.num_repeats)
        ],
        "repeats": repeat_summaries,
        "verbose": spec.verbose,
        "metrics": metrics,
        **prediction_stats,
        "primary_metric": primary_metric,
        "primary_score": primary_score,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
