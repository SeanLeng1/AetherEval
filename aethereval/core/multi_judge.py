"""Reuse one generation set across judges, with independently resumable scores."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .io import ensure_dir, model_output_name, run_output_dir, write_json
from .run_summary import build_run_summary, load_task_summaries, phase_name
from .task_register import BENCHMARKS_DIR, discover_tasks, load_task, parse_task_names


def validate_judge_models(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "judge_models must be a comma-separated string or nonempty list"
        )
    if any(not isinstance(model, str) or not model.strip() for model in value):
        raise ValueError("judge_models must contain nonempty model names")
    models = [model.strip() for model in value]
    if len(set(models)) != len(models):
        raise ValueError("judge_models must not contain duplicates")
    return models


def judge_directory(run_root: Path, model: str) -> Path:
    # Full model identity prevents collisions between paths with the same suffix.
    suffix = hashlib.sha256(model.encode()).hexdigest()[:8]
    return run_root / "judges" / f"{model_output_name(model)}-{suffix}"


def _generation_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # Judge scores, parsed judgments and judge metadata do not identify
            # the generated answer; they may differ across independent caches.
            generation = {
                key: row.get(key)
                for key in (
                    "sample_id",
                    "gen_idx",
                    "prompt",
                    "generation",
                    "gold",
                    "error",
                )
            }
            digest.update(
                json.dumps(generation, sort_keys=True, ensure_ascii=False).encode()
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _copy_task(source: Path, target: Path, *, replace: bool = False) -> None:
    """Seed a judge cache from saved generations, retaining existing judgments."""
    predictions = sorted(source.rglob("predictions.jsonl"))
    if not predictions:
        raise ValueError(f"No saved generations found in {source}")
    # Check every repeat before writing any of this task's cache.
    for path in predictions:
        destination = target / path.relative_to(source)
        if destination.exists() and not replace:
            if _generation_digest(destination) != _generation_digest(path):
                raise ValueError(
                    f"Judge cache generations differ from {source}: {destination}; "
                    "use a new run directory or --overwrite"
                )
    for path in sorted(source.rglob("run_config.json")):
        destination = target / path.relative_to(source)
        if destination.exists() and not replace:
            saved, requested = (
                json.loads(destination.read_text()),
                json.loads(path.read_text()),
            )
            keys = (
                "model",
                "model_name",
                "generation_config",
                "task_fingerprint",
                "protocol",
                "num_repeats",
            )
            if any(saved.get(key) != requested.get(key) for key in keys):
                raise ValueError(
                    f"Judge cache generation settings differ: {destination}"
                )
    for name in ("predictions.jsonl", "run_config.json", "summary.json"):
        for path in source.rglob(name):
            destination = target / path.relative_to(source)
            if replace or not destination.exists():
                ensure_dir(destination.parent)
                shutil.copy2(path, destination)


def run_multi_judge(
    *,
    options: dict[str, Any],
    overwrite: bool,
    backend: Any,
    generate_only: bool,
    eval_only: bool,
) -> dict[str, Any]:
    from .runner import _run_phase

    metric_options = dict(options.get("metric_options") or {})
    models = validate_judge_models(metric_options.pop("judge_models"))
    if metric_options.get("judge_model") is not None:
        raise ValueError("judge_model and judge_models are mutually exclusive")
    options = {
        **options,
        "metric_options": {**metric_options, "judge_model": models[0]},
    }
    task_root = options.get("benchmarks_dir") or BENCHMARKS_DIR
    available = sorted(discover_tasks(task_root))
    selected = parse_task_names(options["tasks"], available)
    judged = [
        name
        for name in selected
        if getattr(load_task(name, task_root).metrics_module, "USES_LLM_JUDGE", False)
    ]
    if not judged:
        raise ValueError("judge_models requires at least one LLM-judge task")
    ordinary = [name for name in selected if name not in judged]
    run_root = run_output_dir(
        options["output_dir"],
        options["model"],
        options["run_id"],
        options["model_name"],
    )
    run_id = options["run_id"] or model_output_name(
        options["model"], options["model_name"]
    )
    phase = phase_name(generate_only=generate_only, eval_only=eval_only)

    if not eval_only:
        generated = _run_phase(
            **options,
            overwrite=overwrite,
            backend=backend,
            generate_only=True,
            eval_only=False,
            rescore_existing=True,
        )
        if generate_only:
            return generated

    if ordinary:
        _run_phase(
            **{**options, "tasks": ",".join(ordinary)},
            overwrite=False,
            backend=backend,
            generate_only=False,
            eval_only=True,
            rescore_existing=eval_only,
        )
    ordinary_summaries = load_task_summaries(run_root, allowed_tasks=available)
    ordinary_summaries = {
        name: summary
        for name, summary in ordinary_summaries.items()
        if not getattr(
            load_task(name, task_root).metrics_module, "USES_LLM_JUDGE", False
        )
    }
    reports = {}
    result = {
        "run_id": run_id,
        "model": options["model"],
        "model_name": model_output_name(options["model"], options["model_name"]),
        "selected_tasks": selected,
        "phase": phase,
        "judge_models": models,
        "judges": reports,
        "evaluation_complete": False,
    }
    write_json(run_root / "run_summary.json", result)
    for model in models:
        profile_root = judge_directory(run_root, model)
        for name in judged:
            _copy_task(run_root / name, profile_root / name, replace=overwrite)
        # Only the selected judge tasks are evaluated again. Each phase closes
        # its local judge before the next model is loaded.
        judged_result = _run_phase(
            **{
                **options,
                "tasks": ",".join(judged),
                "metric_options": {**metric_options, "judge_model": model},
            },
            overwrite=False,
            backend=backend,
            generate_only=False,
            eval_only=True,
            rescore_existing=eval_only,
            run_root_override=profile_root,
        )
        for name in ordinary_summaries:
            source, target = run_root / name, profile_root / name
            ensure_dir(target)
            for filename in ("summary.json", "run_config.json"):
                if (source / filename).exists():
                    shutil.copy2(source / filename, target / filename)
        profile = build_run_summary(
            run_root=profile_root,
            run_id=run_id,
            selected_tasks=selected,
            model=options["model"],
            model_name=model_output_name(options["model"], options["model_name"]),
            backend=judged_result["backend"],
            phase=phase,
            task_summaries={**ordinary_summaries, **judged_result["results"]},
        )
        reports[model] = {"output_dir": str(profile_root), **profile}
        result["backend"] = judged_result["backend"]
        result["evaluation_complete"] = len(reports) == len(models) and all(
            report["results"].get(name, {}).get("evaluation_complete")
            for report in reports.values()
            for name in selected
        )
        write_json(run_root / "run_summary.json", result)
    return result
