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
            register_rlla_model(spec.model)

        if spec.run_generation:
            from bfcl_eval._llm_response_generation import main as generation_main

            with _filter_bfcl_prints(not spec.verbose), _cap_bfcl_thread_pool(
                spec.num_threads
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
    primary_metric = "avg_acc"
    primary_score = float(metrics.get(primary_metric, 0.0))
    _write_summary(out, spec, metrics, primary_metric, primary_score)
    return ExternalResult(metrics, primary_metric, primary_score, result_dir, score_dir)


def _read_overall_csv(score_dir: Path) -> dict[str, float] | None:
    """Parse BFCL's leaderboard ``data_overall.csv`` -> Table 1 overall accuracies."""
    csv_path = score_dir / "data_overall.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]  # single registered model

    def pct(col: str) -> float | None:
        v = row.get(col)
        if v is None:
            return None
        try:
            return float(str(v).replace("%", "").strip())
        except ValueError:
            return None  # "N/A" (category not run)

    # Exact BFCL leaderboard columns -> GDPO Table 1 accuracies.
    mapping = {
        "non_live_overall_acc": "Non-Live AST Acc",
        "live_overall_acc": "Live Acc",
        "multi_turn_overall_acc": "Multi Turn Acc",
        "overall_acc": "Overall Acc",
    }
    out: dict[str, float] = {}
    for metric, col in mapping.items():
        val = pct(col)
        if val is not None:
            out[metric] = val
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
    if overalls:
        metrics["avg_acc"] = sum(overalls) / len(overalls)
    return metrics


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


def _write_summary(
    out: Path,
    spec: ExternalRunSpec,
    metrics: dict[str, float],
    primary_metric: str,
    primary_score: float,
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
        "primary_metric": primary_metric,
        "primary_score": primary_score,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
