from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from .task_defaults import resolve_task_default_gen
from .types import TaskBundle, TaskSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"


def discover_tasks(benchmarks_dir: Path | None = None) -> dict[str, TaskSpec]:
    root = benchmarks_dir or BENCHMARKS_DIR
    tasks: dict[str, TaskSpec] = {}
    if not root.exists():
        return tasks

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_module_path = task_dir / "task.py"
        metrics_module_path = task_dir / "metrics.py"
        if not (task_module_path.exists() and metrics_module_path.exists()):
            continue
        name = task_dir.name
        tasks[name] = TaskSpec(
            name=name,
            task_dir=task_dir,
            task_module_path=task_module_path,
            metrics_module_path=metrics_module_path,
        )

    return tasks


def list_tasks(benchmarks_dir: Path | None = None) -> list[str]:
    return sorted(discover_tasks(benchmarks_dir).keys())


def list_task_default_gens(benchmarks_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    tasks = discover_tasks(benchmarks_dir)
    resolved: dict[str, dict[str, Any]] = {}
    for task_name in sorted(tasks.keys()):
        spec = tasks[task_name]
        task_module = _load_module_from_path(
            f"aethereval_task_defaults_{task_name}",
            spec.task_module_path,
        )
        _validate_task_contract(task_module)
        fallback_default_gen = getattr(task_module, "DEFAULT_GEN", {})
        resolved[task_name] = resolve_task_default_gen(task_name, fallback_default_gen)
    return resolved


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_task_contract(module: ModuleType) -> None:
    required_attrs = ["TASK_NAME", "DATA_FILE"]
    required_funcs = ["load_samples", "build_prompt"]

    missing_attrs = [name for name in required_attrs if not hasattr(module, name)]
    missing_funcs = [
        name
        for name in required_funcs
        if not hasattr(module, name) or not callable(getattr(module, name))
    ]

    if missing_attrs or missing_funcs:
        parts = []
        if missing_attrs:
            parts.append(f"missing attrs: {', '.join(missing_attrs)}")
        if missing_funcs:
            parts.append(f"missing funcs: {', '.join(missing_funcs)}")
        raise ValueError(
            f"Task module '{module.__name__}' contract invalid ({'; '.join(parts)})"
        )

    task_name = getattr(module, "TASK_NAME")
    data_file = getattr(module, "DATA_FILE")

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("TASK_NAME must be a non-empty string")
    if not isinstance(data_file, str) or not data_file.endswith(".jsonl"):
        raise ValueError("DATA_FILE must be a .jsonl path under the task folder")
    if hasattr(module, "DEFAULT_GEN") and not isinstance(getattr(module, "DEFAULT_GEN"), dict):
        raise ValueError("DEFAULT_GEN must be a dict when provided")


def _validate_metrics_contract(module: ModuleType) -> None:
    required_funcs = ["score_generation", "aggregate"]
    missing_funcs = [
        name
        for name in required_funcs
        if not hasattr(module, name) or not callable(getattr(module, name))
    ]
    if missing_funcs:
        raise ValueError(
            f"Metrics module '{module.__name__}' contract invalid "
            f"(missing funcs: {', '.join(missing_funcs)})"
        )


def load_task(task_name: str, benchmarks_dir: Path | None = None) -> TaskBundle:
    tasks = discover_tasks(benchmarks_dir)
    if task_name not in tasks:
        available = ", ".join(sorted(tasks.keys()))
        raise KeyError(f"Unknown task '{task_name}'. Available: {available}")

    spec = tasks[task_name]
    task_module = _load_module_from_path(
        f"aethereval_task_{task_name}",
        spec.task_module_path,
    )
    metrics_module = _load_module_from_path(
        f"aethereval_metrics_{task_name}",
        spec.metrics_module_path,
    )
    _validate_task_contract(task_module)
    fallback_default_gen = getattr(task_module, "DEFAULT_GEN", {})
    task_module.DEFAULT_GEN = resolve_task_default_gen(task_name, fallback_default_gen)
    _validate_metrics_contract(metrics_module)
    return TaskBundle(spec=spec, task_module=task_module, metrics_module=metrics_module)
