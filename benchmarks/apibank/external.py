"""External-benchmark API for API-Bank (GD2PO) in AetherEval.

API-Bank is wrapped as an *external benchmark* (same ``ExternalRunSpec`` /
``ExternalResult`` / ``run`` shape as ``benchmarks/bfcl``): it owns the reference's
deterministic generation setup (greedy: temperature=0, top_p=1, seed=42,
max_tokens=4096, max_model_len=4096) and the reference ``result.json`` /
``score_reward.json`` / ``leaderboard.json`` layout, so AetherEval's native
single-shot ``task.py``/``metrics.py`` contract is bypassed. Generation runs through
AetherEval's backend abstraction (sglang default, vllm as in the reference).

    spec = ExternalRunSpec(model="rlla-qwen", model_path="/ckpt", output_dir=Path("out"))
    result = run(spec)  # -> ExternalResult(metrics, primary_metric, primary_score)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aethereval.backends.factory import create_backend
from aethereval.core.types import GenerationInput

from .scoring import aggregate_scores, parse_assistant_output, score_record

DATA_DIR = Path(__file__).resolve().parent / "data"
ALL_LEVELS = ("1", "2", "3")


@dataclass
class ExternalRunSpec:
    model: str                              # HF id, or any name paired with model_path
    output_dir: Path                        # run dir; result/score/leaderboard/summary go under it
    model_path: str | None = None           # local checkpoint dir (None => load `model` from HF)
    levels: list[str] = field(default_factory=lambda: list(ALL_LEVELS))
    backend: str = "sglang"                 # tmux0 container ships sglang; reference used vllm
    dp_size: int = 1
    tp_size: int = 1
    gpu_memory_utilization: float = 0.6     # vllm engine arg (reference default)
    mem_fraction_static: float = 0.8        # sglang engine arg
    seed: int = 42
    max_tokens: int = 4096
    max_model_len: int = 4096
    run_generation: bool = True
    run_evaluation: bool = True


@dataclass
class ExternalResult:
    metrics: dict[str, Any]
    primary_metric: str
    primary_score: float
    result_path: Path
    score_path: Path


def load_level(level: str) -> list[dict[str, Any]]:
    with open(DATA_DIR / f"level-{level}-api_processed.json", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def _engine_kwargs(spec: ExternalRunSpec) -> dict[str, Any]:
    if spec.backend == "vllm":
        return {
            "max_model_len": spec.max_model_len,
            "seed": spec.seed,
            "trust_remote_code": True,
            "gpu_memory_utilization": spec.gpu_memory_utilization,
        }
    return {
        "context_length": spec.max_model_len,
        "random_seed": spec.seed,
        "trust_remote_code": True,
        "mem_fraction_static": spec.mem_fraction_static,
    }


def _generate(spec: ExternalRunSpec, out: Path) -> None:
    inputs: list[GenerationInput] = []
    data_by_key: dict[str, dict[str, Any]] = {}
    for level in spec.levels:
        for idx, data in enumerate(load_level(level)):
            key = f"Level{level}_{idx}"
            data_by_key[key] = data
            # Reference prompt: llm.chat([system, user]) == chat template + generation prompt.
            inputs.append(GenerationInput(
                sample_id=key,
                prompt=[
                    {"role": "system", "content": data["system"]},
                    {"role": "user", "content": data["user"]},
                ],
            ))

    backend = create_backend(
        backend_name=spec.backend,
        model=spec.model_path or spec.model,
        dp_size=spec.dp_size,
        tensor_parallel_size=spec.tp_size,
        model_kwargs=_engine_kwargs(spec),
    )
    # Reference greedy sampling params.
    gen_cfg = {
        "max_new_tokens": spec.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": spec.seed,
    }
    outputs = backend.generate(inputs, gen_cfg)
    backend.close()

    results: dict[str, Any] = {}
    errors: dict[str, Any] = {}
    for output in outputs:
        data = data_by_key[output.sample_id]
        if output.error is not None:
            errors[output.sample_id] = {"data": data, "error": output.error}
            continue
        raw_output = output.generations[0].strip()
        thought, tool_calls = parse_assistant_output(raw_output)
        results[output.sample_id] = {
            "data": data,
            "raw_output": raw_output,
            "thought": thought,
            "tool_calls": tool_calls,
        }

    _dump_json(results, out / "result.json")
    _dump_json(errors, out / "error.json")


def _evaluate(out: Path, model_key: str) -> dict[str, Any]:
    with open(out / "result.json", encoding="utf-8") as f:
        results = json.load(f)

    scores = {key: score_record(record) for key, record in results.items()}
    _dump_json(scores, out / "score_reward.json")

    record = aggregate_scores(scores)
    _dump_json({model_key: record}, out / "leaderboard.json")
    return record


def _write_summary(
    out: Path,
    spec: ExternalRunSpec,
    metrics: dict[str, Any],
    primary_metric: str,
    primary_score: float,
) -> None:
    summary = {
        "benchmark": "apibank",
        "external": True,
        "model": spec.model,
        "model_path": spec.model_path,
        "backend": spec.backend,
        "levels": list(spec.levels),
        "seed": spec.seed,
        "max_tokens": spec.max_tokens,
        "max_model_len": spec.max_model_len,
        "metrics": metrics,
        "primary_metric": primary_metric,
        "primary_score": primary_score,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def run(spec: ExternalRunSpec) -> ExternalResult:
    out = Path(spec.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if spec.run_generation:
        _generate(spec, out)

    metrics: dict[str, Any] = {}
    primary_metric = "overall_acc"
    primary_score = 0.0
    if spec.run_evaluation:
        metrics = _evaluate(out, spec.model.replace("/", "_"))
        if metrics[primary_metric] is not None:
            primary_score = float(metrics[primary_metric])
        _write_summary(out, spec, metrics, primary_metric, primary_score)

    return ExternalResult(
        metrics=metrics,
        primary_metric=primary_metric,
        primary_score=primary_score,
        result_path=out / "result.json",
        score_path=out / "score_reward.json",
    )
