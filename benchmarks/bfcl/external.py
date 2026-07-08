"""External-benchmark API for BFCL-v3 in AetherEval."""

import builtins
import contextlib
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .register import register_rlla_model

# BFCL category collections -> GDPO Table 1 columns.
_COLLECTIONS = ["non_live", "live", "multi_turn"]

# ToolRL output-format patterns (GDPO Table 1 "Correct Format"): well-formed iff the
# response matches one of the canonical think / tool_call / response shapes.
_FMT_PATTERNS = [
    r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$",
    r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$",
    r"^<think>.*?</think>\n<response>.*?</response>$",
    r"^<think>.*?</think>$",
]
_NOISY_BFCL_MESSAGES = {
    "Empty response from the model. Proceed to next turn.",
    "Failed to decode the model response. Proceed to next turn.",
}


@dataclass
class ExternalRunSpec:
    model: str  # registry name = HF id, or any name with model_path
    output_dir: Path  # AetherEval run dir; result/ + score/ go under it
    model_path: str | None = None  # local checkpoint dir (None => load `model` from HF)
    categories: list[str] = field(default_factory=lambda: ["all"])
    backend: str = "sglang"  # tmux0 container ships sglang
    num_gpus: int = 1
    num_threads: int = 16
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.001  # near-greedy, BFCL tool-calling default
    max_tokens: int = 4096
    max_context_length: int | None = None
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    verbose: bool = False
    allow_overwrite: bool = True
    run_generation: bool = True
    run_evaluation: bool = True


@dataclass
class ExternalResult:
    metrics: dict[str, float]
    primary_metric: str
    primary_score: float
    result_dir: Path
    score_dir: Path


def _gen_args(spec: ExternalRunSpec, result_dir: Path) -> SimpleNamespace:
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
        skip_server_setup=False,
        local_model_path=spec.model_path,
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
    }


@contextlib.contextmanager
def _cap_bfcl_thread_pool(max_workers: int):
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


def _server_command_with_context(
    command,
    max_context_length: int | None,
):
    if max_context_length is None:
        return command
    if not isinstance(command, (list, tuple)):
        return command

    patched = list(command)
    if "sglang.launch_server" in patched and "--context-length" not in patched:
        patched.extend(["--context-length", str(max_context_length)])
    elif patched[:2] == ["vllm", "serve"] and "--max-model-len" not in patched:
        patched.extend(["--max-model-len", str(max_context_length)])

    return type(command)(patched) if isinstance(command, tuple) else patched


@contextlib.contextmanager
def _patch_bfcl_server_context(max_context_length: int | None):
    if max_context_length is None:
        yield
        return

    from bfcl_eval.model_handler.local_inference import base_oss_handler

    original = base_oss_handler.subprocess.Popen

    def context_popen(*args, **kwargs):
        if args:
            args = (
                _server_command_with_context(args[0], max_context_length),
                *args[1:],
            )
        elif "args" in kwargs:
            kwargs["args"] = _server_command_with_context(
                kwargs["args"],
                max_context_length,
            )
        return original(*args, **kwargs)

    base_oss_handler.subprocess.Popen = context_popen
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


def run(spec: ExternalRunSpec) -> ExternalResult:
    out = Path(spec.output_dir).resolve()
    result_dir = out / "result"
    score_dir = out / "score"
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    with _temporary_env(_handler_env(spec)):
        if spec.run_generation or spec.run_evaluation:
            register_rlla_model(spec.model, project_root=str(out))

        if spec.run_generation:
            from bfcl_eval._llm_response_generation import main as generation_main

            with (
                _filter_bfcl_prints(not spec.verbose),
                _cap_bfcl_thread_pool(spec.num_threads),
                _patch_bfcl_server_context(spec.max_context_length),
            ):
                generation_main(_gen_args(spec, result_dir))

        if spec.run_generation or spec.run_evaluation:
            _raise_on_inference_errors(result_dir, spec.model)

        if spec.run_evaluation:
            from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

            # 4 positional args work on BFCL v3 (no partial_eval) and v4 (defaulted).
            evaluation_main(
                [spec.model], list(spec.categories), str(result_dir), str(score_dir)
            )

    metrics = parse_scores(score_dir, spec.model)
    fmt = compute_format_rate(result_dir, spec.model)
    if fmt is not None:
        metrics["correct_format"] = fmt
    prediction_stats = write_predictions_jsonl(
        out=out,
        result_dir=result_dir,
        score_dir=score_dir,
        model=spec.model,
    )
    primary_metric = "OverallAcc"
    primary_score = float(metrics.get(primary_metric, 0.0))
    _write_summary(out, spec, metrics, primary_metric, primary_score, prediction_stats)
    return ExternalResult(metrics, primary_metric, primary_score, result_dir, score_dir)


def _read_overall_csv(score_dir: Path) -> dict[str, float] | None:
    """Parse BFCL's leaderboard ``data_overall.csv`` into ToolRL report metrics."""
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
        "OverallAcc": ("Overall Acc",),
        "Non-LiveASTAcc": ("Non-Live AST Acc",),
        "Non-LiveExecAcc": ("Non-Live Exec Acc",),
        "LiveAcc": ("Live Acc",),
        "MultiTurnAcc": ("Multi Turn Acc",),
        "RelevanceDetection": ("Relevance Detection",),
        "IrrelevanceDetection": ("Irrelevance Detection",),
    }
    out: dict[str, float] = {}
    for metric, columns in report_mapping.items():
        value = pct(*columns)
        if value is not None:
            out[metric] = value

    alias_mapping = {
        "overall_acc": "OverallAcc",
        "non_live_ast_acc": "Non-LiveASTAcc",
        "non_live_exec_acc": "Non-LiveExecAcc",
        "live_acc": "LiveAcc",
        "multi_turn_acc": "MultiTurnAcc",
        "relevance_detection": "RelevanceDetection",
        "irrelevance_detection": "IrrelevanceDetection",
    }
    for alias, source in alias_mapping.items():
        if source in out:
            out[alias] = out[source]

    legacy_mapping = {
        "non_live_overall_acc": "Non-LiveASTAcc",
        "live_overall_acc": "LiveAcc",
        "multi_turn_overall_acc": "MultiTurnAcc",
        "avg_acc": "OverallAcc",
    }
    for alias, source in legacy_mapping.items():
        if source in out:
            out[alias] = out[source]
    return out or None


def _read_category_jsons(score_dir: Path, model: str) -> dict[str, float]:
    """Fallback: aggregate per-category ``*_score.json`` (first line = {accuracy,...})."""
    model_dir = score_dir / model.replace("/", "_")
    score_files = list(model_dir.rglob("*_score.json")) if model_dir.exists() else []
    if not score_files:
        return {}

    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING

    per_cat: dict[str, float] = {}
    for jf in score_files:
        cat = jf.stem.split("_score")[0]
        cat = cat.split("_", 2)[-1] if cat.startswith("BFCL") else cat
        with open(jf) as f:
            head = json.loads(f.readline())
        if isinstance(head, dict) and "accuracy" in head:
            per_cat[cat] = float(head["accuracy"]) * 100.0

    out: dict[str, float] = {}
    for coll in _COLLECTIONS:
        cats = TEST_COLLECTION_MAPPING.get(coll, [])
        vals = [per_cat[c] for c in cats if c in per_cat]
        if vals:
            out[f"{coll}_overall_acc"] = sum(vals) / len(vals)
    out.update({f"cat/{k}": v for k, v in per_cat.items()})
    return out


def parse_scores(score_dir: Path, model: str) -> dict[str, float]:
    metrics = _read_overall_csv(score_dir) or {}
    if not metrics:
        metrics = _read_category_jsons(score_dir, model)
    overalls = [
        metrics[f"{c}_overall_acc"]
        for c in _COLLECTIONS
        if f"{c}_overall_acc" in metrics
    ]
    if "OverallAcc" in metrics:
        metrics["avg_acc"] = metrics["OverallAcc"]
    elif overalls:
        metrics["avg_acc"] = sum(overalls) / len(overalls)
        metrics["OverallAcc"] = metrics["avg_acc"]
        metrics["overall_acc"] = metrics["avg_acc"]
    if "Non-LiveASTAcc" not in metrics and "non_live_overall_acc" in metrics:
        metrics["Non-LiveASTAcc"] = metrics["non_live_overall_acc"]
        metrics["non_live_ast_acc"] = metrics["non_live_overall_acc"]
    if "LiveAcc" not in metrics and "live_overall_acc" in metrics:
        metrics["LiveAcc"] = metrics["live_overall_acc"]
        metrics["live_acc"] = metrics["live_overall_acc"]
    if "MultiTurnAcc" not in metrics and "multi_turn_overall_acc" in metrics:
        metrics["MultiTurnAcc"] = metrics["multi_turn_overall_acc"]
        metrics["multi_turn_acc"] = metrics["multi_turn_overall_acc"]
    return metrics


def _is_allowed_zero_score_error(error: str) -> bool:
    return "BFCL prompt exceeds max context length:" in error


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
                values = result if isinstance(result, list) else [result]
                errors = [
                    str(value)
                    for value in values
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


def compute_format_rate(result_dir: Path, model: str) -> float | None:
    """Fraction (%) of generated responses matching the ToolRL output format."""
    model_dir = result_dir / model.replace("/", "_")
    total = ok = 0
    for jf in model_dir.rglob("*_result.json"):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    resp = json.loads(line).get("result", "")
                except json.JSONDecodeError:
                    continue
                if isinstance(resp, list):  # multi-turn -> per-turn strings
                    resp = "\n".join(str(x) for x in resp)
                total += 1
                text = str(resp).strip()
                if any(re.search(p, text, re.DOTALL) for p in _FMT_PATTERNS):
                    ok += 1
    return 100.0 * ok / total if total else None


def _as_generation_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _result_error(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("Error during inference:"):
        return value
    return None


def _result_file_category(path: Path) -> str:
    name = path.name
    if name.endswith("_result.json"):
        name = name[: -len("_result.json")]
    if name.startswith("BFCL_"):
        name = name[len("BFCL_") :]
    return name


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
) -> dict[str, int | str]:
    predictions_path = out / "predictions.jsonl"
    model_dir = result_dir / model.replace("/", "_")
    score_model_dir = score_dir / model.replace("/", "_")
    result_files = sorted(model_dir.rglob("*_result.json")) if model_dir.exists() else []

    total_records = 0
    scored_records = 0
    with predictions_path.open("w", encoding="utf-8") as f:
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
                        "gen_idx": 0,
                        "prompt": _prompt_from_bfcl_record(raw_record),
                        "generation": _as_generation_text(raw_result),
                        "score": score,
                        "is_pass": is_pass,
                        "parsed": raw_result,
                        "gold": None,
                        "error": _result_error(raw_result),
                        "meta": {
                            "benchmark": "bfcl",
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
) -> None:
    summary = {
        "benchmark": "bfcl",
        "external": True,
        "model": spec.model,
        "model_path": spec.model_path,
        "backend": spec.backend,
        "categories": list(spec.categories),
        "num_gpus": spec.num_gpus,
        "num_threads": spec.num_threads,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "max_context_length": spec.max_context_length,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "repetition_penalty": spec.repetition_penalty,
        "verbose": spec.verbose,
        "metrics": metrics,
        **prediction_stats,
        "primary_metric": primary_metric,
        "primary_score": primary_score,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
