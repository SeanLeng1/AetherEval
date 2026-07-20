import json
from pathlib import Path
from typing import Any, Iterable

from .io import write_json


def phase_name(*, generate_only: bool, eval_only: bool) -> str:
    if generate_only:
        return "generate_only"
    if eval_only:
        return "eval_only"
    return "generate_and_eval"


def load_task_summaries(
    run_root: Path,
    *,
    skip_tasks: Iterable[str] = (),
    allowed_tasks: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not run_root.exists():
        return {}

    skipped = set(skip_tasks)
    allowed = set(allowed_tasks) if allowed_tasks is not None else None
    summaries: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(run_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name in skipped:
            continue
        if allowed is not None and task_dir.name not in allowed:
            continue

        summary_path = task_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        if not isinstance(summary, dict):
            raise ValueError(f"Existing summary must be a JSON object: {summary_path}")
        summaries[task_dir.name] = summary
    return summaries


def _mean_numeric_metrics(
    task_summaries: dict[str, dict[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for summary in task_summaries.values():
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Task summary metrics must be a dict")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped.setdefault(str(key), []).append(float(value))
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def _mean_primary_score(task_summaries: dict[str, dict[str, Any]]) -> float | None:
    values = [
        float(summary["primary_score"])
        for summary in task_summaries.values()
        if isinstance(summary.get("primary_score"), (int, float))
    ]
    return sum(values) / len(values) if values else None


def build_run_summary(
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
    summary = {
        "run_id": run_id,
        "selected_tasks": selected_tasks,
        "tasks": sorted(task_summaries),
        "model": model,
        "model_name": model_name,
        "backend": backend,
        "phase": phase,
        "results": task_summaries,
        "primary_scores": {
            task_name: {
                "metric": task_summary.get("primary_metric"),
                "score": task_summary.get("primary_score"),
            }
            for task_name, task_summary in task_summaries.items()
        },
        "primary_score_aggregate": _mean_primary_score(task_summaries),
        "summary": {
            "num_tasks": len(task_summaries),
            "metrics": _mean_numeric_metrics(task_summaries),
        },
    }
    write_json(run_root / "run_summary.json", summary)
    return summary
