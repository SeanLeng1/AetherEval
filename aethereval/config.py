import ast
from pathlib import Path
from typing import Any

from aethereval.backends.factory import SUPPORTED_BACKENDS


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
    except (ValueError, SyntaxError):
        return text


def _parse_key_value_args(values: Any, flag_name: str) -> dict[str, Any]:
    if not values:
        return {}
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{flag_name} must be used as repeated key=value entries")

    parsed: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid {flag_name} '{raw}', expected key=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid {flag_name} '{raw}', empty key")
        parsed[key] = _parse_scalar(value)
    return parsed


def _parse_vllm_args(values: Any) -> dict[str, Any]:
    return _parse_key_value_args(values, "--vllm-arg")


def _parse_sglang_args(values: Any) -> dict[str, Any]:
    return _parse_key_value_args(values, "--sglang-arg")


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
    model_name = _pick(
        getattr(args, "model_name", None),
        _cfg_get(cfg, "model_name", "run"),
    )
    backend = str(
        _pick(
            getattr(args, "backend", None),
            _cfg_get(cfg, "backend", "runtime"),
            "vllm",
        )
    ).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends: "
            f"{', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    tasks_raw = _pick(args.tasks, _cfg_get(cfg, "tasks", "run"), "all")
    if isinstance(tasks_raw, (list, tuple)):
        tasks = ",".join(str(x) for x in tasks_raw)
    else:
        tasks = str(tasks_raw)

    output_dir = _pick(args.output_dir, _cfg_get(cfg, "output_dir", "run"), "outputs")
    run_id = _pick(args.run_id, _cfg_get(cfg, "run_id", "run"))
    overwrite = bool(_pick(args.overwrite, _cfg_get(cfg, "overwrite", "run"), False))
    inspect = bool(
        _pick(getattr(args, "inspect", None), _cfg_get(cfg, "inspect", "run"), False)
    )
    generate_only = bool(
        _pick(
            getattr(args, "generate_only", None),
            _cfg_get(cfg, "generate_only", "run"),
            False,
        )
    )
    eval_only = bool(
        _pick(
            getattr(args, "eval_only", None),
            _cfg_get(cfg, "eval_only", "run"),
            False,
        )
    )
    if generate_only and eval_only:
        raise ValueError("generate_only and eval_only are mutually exclusive")
    if eval_only and overwrite:
        raise ValueError("eval_only cannot be combined with overwrite")

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
        "temperature": _pick(
            args.temperature, _cfg_get(cfg, "temperature", "generation")
        ),
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
    metric_options = {
        "rm_model_path": _pick(
            getattr(args, "rm_model_path", None),
            _cfg_get(cfg, "rm_model_path", "metrics"),
        ),
        "cm_model_path": _pick(
            getattr(args, "cm_model_path", None),
            _cfg_get(cfg, "cm_model_path", "metrics"),
        ),
        "rm_batch_size": _pick(
            getattr(args, "rm_batch_size", None),
            _cfg_get(cfg, "rm_batch_size", "metrics"),
        ),
        "rm_max_length": _pick(
            getattr(args, "rm_max_length", None),
            _cfg_get(cfg, "rm_max_length", "metrics"),
        ),
        "rm_dtype": _pick(
            getattr(args, "rm_dtype", None),
            _cfg_get(cfg, "rm_dtype", "metrics"),
        ),
        "rm_trust_remote_code": _pick(
            getattr(args, "rm_trust_remote_code", None),
            _cfg_get(cfg, "rm_trust_remote_code", "metrics"),
        ),
        "judge_model": _pick(
            getattr(args, "judge_model", None),
            _cfg_get(cfg, "judge_model", "metrics"),
        ),
        "judge_base_url": _pick(
            getattr(args, "judge_base_url", None),
            _cfg_get(cfg, "judge_base_url", "metrics"),
        ),
        "judge_api_key_env": _pick(
            getattr(args, "judge_api_key_env", None),
            _cfg_get(cfg, "judge_api_key_env", "metrics"),
        ),
        "judge_workers": _pick(
            getattr(args, "judge_workers", None),
            _cfg_get(cfg, "judge_workers", "metrics"),
        ),
        "judge_timeout": _pick(
            getattr(args, "judge_timeout", None),
            _cfg_get(cfg, "judge_timeout", "metrics"),
        ),
        "judge_max_retries": _pick(
            getattr(args, "judge_max_retries", None),
            _cfg_get(cfg, "judge_max_retries", "metrics"),
        ),
        "judge_repeats": _pick(
            getattr(args, "judge_repeats", None),
            _cfg_get(cfg, "judge_repeats", "metrics"),
        ),
    }
    metric_options = {k: v for k, v in metric_options.items() if v is not None}

    vllm_kwargs = {
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
    if cfg_extra_model_kwargs is not None and not isinstance(
        cfg_extra_model_kwargs, dict
    ):
        raise ValueError("vllm.extra_model_kwargs must be a mapping/object")
    if isinstance(cfg_extra_model_kwargs, dict):
        vllm_kwargs.update(cfg_extra_model_kwargs)

    cli_extra = _parse_vllm_args(getattr(args, "vllm_arg", None))
    vllm_kwargs.update(cli_extra)
    vllm_kwargs = {k: v for k, v in vllm_kwargs.items() if v is not None}

    sglang_kwargs = {
        "mem_fraction_static": _pick(
            getattr(args, "mem_fraction_static", None),
            _cfg_get(cfg, "mem_fraction_static", "sglang"),
        ),
        "context_length": _pick(
            getattr(args, "context_length", None),
            _cfg_get(cfg, "context_length", "sglang"),
        ),
        "generation_batch_size": _pick(
            getattr(args, "sglang_generation_batch_size", None),
            _cfg_get(cfg, "generation_batch_size", "sglang"),
        ),
        "dtype": _pick(args.dtype, _cfg_get(cfg, "dtype", "sglang")),
    }
    cfg_sglang_extra = _cfg_get(cfg, "extra_model_kwargs", "sglang")
    if cfg_sglang_extra is not None and not isinstance(cfg_sglang_extra, dict):
        raise ValueError("sglang.extra_model_kwargs must be a mapping/object")
    if isinstance(cfg_sglang_extra, dict):
        sglang_kwargs.update(cfg_sglang_extra)

    sglang_cli_extra = _parse_sglang_args(getattr(args, "sglang_arg", None))
    sglang_kwargs.update(sglang_cli_extra)
    sglang_kwargs = {k: v for k, v in sglang_kwargs.items() if v is not None}

    backend_kwargs = vllm_kwargs if backend == "vllm" else sglang_kwargs

    return {
        "model": model,
        "model_name": model_name,
        "backend": backend,
        "tasks": tasks,
        "inspect": inspect,
        "generate_only": generate_only,
        "eval_only": eval_only,
        "output_dir": output_dir,
        "run_id": run_id,
        "overwrite": overwrite,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "gen_overrides": gen_overrides,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_confidence": bootstrap_confidence,
        "metric_options": metric_options,
        "backend_kwargs": backend_kwargs,
        "model_kwargs": backend_kwargs,
        "vllm_kwargs": vllm_kwargs,
        "sglang_kwargs": sglang_kwargs,
    }
