from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _cfg_get(cfg: dict[str, Any], key: str, section: str | None = None) -> Any:
    if section:
        scoped = cfg.get(section)
        if isinstance(scoped, dict) and key in scoped:
            return scoped[key]
    return cfg.get(key)


def _pick(cli_value: Any, cfg_value: Any, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:  # noqa: BLE001
        return text


def _parse_vllm_args(values: Any) -> dict[str, Any]:
    if not values:
        return {}
    if not isinstance(values, (list, tuple)):
        raise ValueError("--vllm-arg must be used as repeated key=value entries")

    parsed: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --vllm-arg '{raw}', expected key=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --vllm-arg '{raw}', empty key")
        parsed[key] = _parse_scalar(value)
    return parsed


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config support. Install requirements first."
        ) from exc

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a mapping/object.")

    return data


def resolve_run_arguments(args: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    model = _pick(args.model, _cfg_get(cfg, "model", "run"))

    tasks_raw = _pick(args.tasks, _cfg_get(cfg, "tasks", "run"), "all")
    if isinstance(tasks_raw, (list, tuple)):
        tasks = ",".join(str(x) for x in tasks_raw)
    else:
        tasks = str(tasks_raw)

    output_dir = _pick(args.output_dir, _cfg_get(cfg, "output_dir", "run"), "outputs")
    run_id = _pick(args.run_id, _cfg_get(cfg, "run_id", "run"))
    overwrite = bool(_pick(args.overwrite, _cfg_get(cfg, "overwrite", "run"), False))
    inspect = bool(_pick(getattr(args, "inspect", None), _cfg_get(cfg, "inspect", "run"), False))

    arg_dp_size = getattr(args, "dp_size", None)
    arg_tp_size = getattr(args, "tp_size", None)

    dp_size = int(_pick(arg_dp_size, _cfg_get(cfg, "dp_size", "runtime"), 1))
    tp_size = int(_pick(arg_tp_size, _cfg_get(cfg, "tp_size", "runtime"), 1))

    gen_overrides = {
        "n": _pick(args.n, _cfg_get(cfg, "n", "generation")),
        "max_new_tokens": _pick(
            args.max_new_tokens,
            _cfg_get(cfg, "max_new_tokens", "generation"),
        ),
        "temperature": _pick(args.temperature, _cfg_get(cfg, "temperature", "generation")),
        "top_p": _pick(args.top_p, _cfg_get(cfg, "top_p", "generation")),
        "top_k": _pick(args.top_k, _cfg_get(cfg, "top_k", "generation")),
        "min_p": _pick(args.min_p, _cfg_get(cfg, "min_p", "generation")),
        "seed": _pick(args.seed, _cfg_get(cfg, "seed", "generation")),
    }

    bootstrap_resamples = int(
        _pick(
            getattr(args, "bootstrap_resamples", None),
            _cfg_get(cfg, "bootstrap_resamples", "metrics"),
            1000,
        )
    )
    bootstrap_seed = int(
        _pick(
            getattr(args, "bootstrap_seed", None),
            _cfg_get(cfg, "bootstrap_seed", "metrics"),
            42,
        )
    )
    bootstrap_confidence = float(
        _pick(
            getattr(args, "bootstrap_confidence", None),
            _cfg_get(cfg, "bootstrap_confidence", "metrics"),
            0.95,
        )
    )

    model_kwargs = {
        "gpu_memory_utilization": _pick(
            args.gpu_memory_utilization,
            _cfg_get(cfg, "gpu_memory_utilization", "vllm"),
        ),
        "max_model_len": _pick(
            args.max_model_len,
            _cfg_get(cfg, "max_model_len", "vllm"),
        ),
        "dtype": _pick(args.dtype, _cfg_get(cfg, "dtype", "vllm")),
    }
    cfg_extra_model_kwargs = _cfg_get(cfg, "extra_model_kwargs", "vllm")
    if cfg_extra_model_kwargs is not None and not isinstance(cfg_extra_model_kwargs, dict):
        raise ValueError("vllm.extra_model_kwargs must be a mapping/object")
    if isinstance(cfg_extra_model_kwargs, dict):
        model_kwargs.update(cfg_extra_model_kwargs)

    cli_extra = _parse_vllm_args(getattr(args, "vllm_arg", None))
    model_kwargs.update(cli_extra)
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    return {
        "model": model,
        "tasks": tasks,
        "inspect": inspect,
        "output_dir": output_dir,
        "run_id": run_id,
        "overwrite": overwrite,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "gen_overrides": gen_overrides,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_confidence": bootstrap_confidence,
        "model_kwargs": model_kwargs,
    }
