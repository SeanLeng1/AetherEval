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


def parse_key_value_args(values: Any, flag_name: str) -> dict[str, Any]:
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
    num_repeats = _pick(
        getattr(args, "num_repeats", None),
        _cfg_get(cfg, "num_repeats", "run"),
    )
    if num_repeats is not None and int(num_repeats) < 1:
        raise ValueError("num_repeats must be >= 1")
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
    if dp_size < 1 or tp_size < 1:
        raise ValueError("runtime dp_size and tp_size must both be >= 1")

    judge_backend = str(
        _pick(
            getattr(args, "judge_backend", None),
            _cfg_get(cfg, "judge_backend", "metrics"),
            "api",
        )
    ).lower()
    if judge_backend not in {"api", "local"}:
        raise ValueError("judge_backend must be 'api' or 'local'")
    raw_judge_dp_size = _pick(
        getattr(args, "judge_dp_size", None),
        _cfg_get(cfg, "judge_dp_size", "metrics"),
    )
    raw_judge_tp_size = _pick(
        getattr(args, "judge_tp_size", None),
        _cfg_get(cfg, "judge_tp_size", "metrics"),
    )
    if raw_judge_dp_size is None and raw_judge_tp_size is None:
        judge_dp_size = 1
        judge_tp_size = dp_size * tp_size
    else:
        judge_dp_size = int(raw_judge_dp_size or 1)
        judge_tp_size = int(raw_judge_tp_size or 1)
    if judge_dp_size < 1 or judge_tp_size < 1:
        raise ValueError("judge dp/tp sizes must both be >= 1")

    cfg_judge_sglang_args = _cfg_get(cfg, "judge_sglang_args", "metrics")
    if cfg_judge_sglang_args is not None and not isinstance(
        cfg_judge_sglang_args, dict
    ):
        raise ValueError("metrics.judge_sglang_args must be a mapping/object")
    judge_sglang_args = dict(cfg_judge_sglang_args or {})
    judge_sglang_args.update(
        parse_key_value_args(
            getattr(args, "judge_sglang_arg", None),
            "--judge-sglang-arg",
        )
    )

    cfg_rm_sglang_args = _cfg_get(cfg, "rm_sglang_args", "metrics")
    if cfg_rm_sglang_args is not None and not isinstance(cfg_rm_sglang_args, dict):
        raise ValueError("metrics.rm_sglang_args must be a mapping/object")
    rm_sglang_args = dict(cfg_rm_sglang_args or {})
    rm_sglang_args.update(
        parse_key_value_args(
            getattr(args, "rm_sglang_arg", None),
            "--rm-sglang-arg",
        )
    )

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
        "enable_thinking": _pick(
            getattr(args, "enable_thinking", None),
            _cfg_get(cfg, "enable_thinking", "generation"),
        ),
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
    metric_keys = (
        "rm_model_path",
        "cm_model_path",
        "rm_max_length",
        "rm_dtype",
        "rm_trust_remote_code",
        "judge_model",
        "judge_base_url",
        "judge_api_key_env",
        "judge_workers",
        "judge_timeout",
        "judge_max_retries",
        "judge_repeats",
        "judge_max_new_tokens",
        "judge_temperature",
        "judge_top_p",
        "judge_enable_thinking",
    )
    metric_options = {
        key: _pick(getattr(args, key, None), _cfg_get(cfg, key, "metrics"))
        for key in metric_keys
    }
    metric_options = {k: v for k, v in metric_options.items() if v is not None}
    if rm_sglang_args:
        metric_options["rm_sglang_args"] = rm_sglang_args
    if judge_backend == "local":
        metric_options.update(
            {
                "judge_backend": "local",
                "judge_dp_size": judge_dp_size,
                "judge_tp_size": judge_tp_size,
                "judge_sglang_args": judge_sglang_args,
            }
        )

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

    cli_extra = parse_key_value_args(getattr(args, "vllm_arg", None), "--vllm-arg")
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
        "dtype": _pick(args.dtype, _cfg_get(cfg, "dtype", "sglang")),
    }
    cfg_sglang_extra = _cfg_get(cfg, "extra_model_kwargs", "sglang")
    if cfg_sglang_extra is not None and not isinstance(cfg_sglang_extra, dict):
        raise ValueError("sglang.extra_model_kwargs must be a mapping/object")
    if isinstance(cfg_sglang_extra, dict):
        sglang_kwargs.update(cfg_sglang_extra)

    sglang_cli_extra = parse_key_value_args(
        getattr(args, "sglang_arg", None),
        "--sglang-arg",
    )
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
        "num_repeats": int(num_repeats) if num_repeats is not None else None,
        "overwrite": overwrite,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "gen_overrides": gen_overrides,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_confidence": bootstrap_confidence,
        "metric_options": metric_options,
        "backend_kwargs": backend_kwargs,
    }
