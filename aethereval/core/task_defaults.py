
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_DEFAULTS_PATH = PROJECT_ROOT / "configs" / "task_defaults.yaml"


@lru_cache(maxsize=1)
def _load_task_default_overrides() -> dict[str, dict[str, Any]]:
    if not TASK_DEFAULTS_PATH.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read configs/task_defaults.yaml. Install requirements first."
        ) from exc

    with TASK_DEFAULTS_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("configs/task_defaults.yaml root must be a mapping/object.")

    parsed: dict[str, dict[str, Any]] = {}
    for task_name, defaults in raw.items():
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_defaults.yaml has an invalid empty task name.")
        if not isinstance(defaults, dict):
            raise ValueError(
                f"task_defaults.yaml entry for '{task_name}' must be a mapping/object."
            )
        parsed[task_name] = dict(defaults)
    return parsed


def resolve_task_default_gen(task_name: str, fallback_default_gen: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fallback_default_gen or {})
    override = _load_task_default_overrides().get(task_name)
    if override:
        merged.update(override)
    return merged


def list_task_default_overrides() -> dict[str, dict[str, Any]]:
    data = _load_task_default_overrides()
    return {name: dict(values) for name, values in data.items()}
