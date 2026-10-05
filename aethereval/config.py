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


def _check_unknown_yaml_keys(
    cfg: dict[str, Any], looked_up: set[tuple[str, str]]
) -> None:
    sections = {section for section, _ in looked_up}
    known = {key for _, key in looked_up}
    unknown = []
    for name, value in cfg.items():
        if name in sections:
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"YAML section '{name}' must be a mapping/object")
            unknown += [
                f"{name}.{key}" for key in value if (name, key) not in looked_up
            ]
        elif name not in known:
            unknown.append(str(name))
    if unknown:
        raise ValueError(f"Unknown YAML config keys: {', '.join(unknown)}")


def resolve_run_arguments(args: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    looked_up: set[tuple[str, str]] = set()

    def get(key: str, section: str) -> Any:
        looked_up.add((section, key))
        return _cfg_get(cfg, key, section)

    model = _pick(args.model, get("model", "run"))
    model_name = _pick(
        getattr(args, "model_name", None),
        get("model_name", "run"),
    )
    backend = str(
        _pick(
            getattr(args, "backend", None),
            get("backend", "runtime"),
            "vllm",
        )
    ).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Supported backends: "
            f"{', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    tasks_raw = _pick(args.tasks, get("tasks", "run"), "all")
    if isinstance(tasks_raw, (list, tuple)):
        tasks = ",".join(str(x) for x in tasks_raw)
    else:
        tasks = str(tasks_raw)

    output_dir = _pick(args.output_dir, get("output_dir", "run"), "outputs")
    run_id = _pick(args.run_id, get("run_id", "run"))
    num_repeats = _pick(
        getattr(args, "num_repeats", None),
        get("num_repeats", "run"),
    )
    if num_repeats is not None and int(num_repeats) < 1:
        raise ValueError("num_repeats must be >= 1")
    overwrite = bool(_pick(args.overwrite, get("overwrite", "run"), False))
    inspect = bool(_pick(getattr(args, "inspect", None), get("inspect", "run"), False))
    cli_generate_only = getattr(args, "generate_only", None)
    cli_eval_only = getattr(args, "eval_only", None)
    cfg_generate_only = get("generate_only", "run")
    cfg_eval_only = get("eval_only", "run")
    if cli_generate_only or cli_eval_only:
        # A phase flag on the CLI replaces the YAML phase instead of combining with it.
        cfg_generate_only = cfg_eval_only = None
    generate_only = bool(_pick(cli_generate_only, cfg_generate_only, False))
    eval_only = bool(_pick(cli_eval_only, cfg_eval_only, False))
    if generate_only and eval_only:
        raise ValueError("generate_only and eval_only are mutually exclusive")
    if eval_only and overwrite:
        raise ValueError("eval_only cannot be combined with overwrite")

    arg_dp_size = getattr(args, "dp_size", None)
    arg_tp_size = getattr(args, "tp_size", None)

    dp_size = int(_pick(arg_dp_size, get("dp_size", "runtime"), 1))
    tp_size = int(_pick(arg_tp_size, get("tp_size", "runtime"), 1))
    if dp_size < 1 or tp_size < 1:
        raise ValueError("runtime dp_size and tp_size must both be >= 1")

    judge_backend = str(
        _pick(
            getattr(args, "judge_backend", None),
            get("judge_backend", "metrics"),
            "api",
        )
    ).lower()
    if judge_backend not in {"api", "local"}:
        raise ValueError("judge_backend must be 'api' or 'local'")
    raw_judge_dp_size = _pick(
        getattr(args, "judge_dp_size", None),
        get("judge_dp_size", "metrics"),
    )
    raw_judge_tp_size = _pick(
        getattr(args, "judge_tp_size", None),
        get("judge_tp_size", "metrics"),
    )
    if raw_judge_dp_size is None and raw_judge_tp_size is None:
        judge_dp_size = 1
        judge_tp_size = dp_size * tp_size
    else:
        judge_dp_size = int(raw_judge_dp_size or 1)
        judge_tp_size = int(raw_judge_tp_size or 1)
    if judge_dp_size < 1 or judge_tp_size < 1:
        raise ValueError("judge dp/tp sizes must both be >= 1")

    cfg_judge_sglang_args = get("judge_sglang_args", "metrics")
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

    cfg_rm_sglang_args = get("rm_sglang_args", "metrics")
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
        "n": _pick(args.n, get("n", "generation")),
        "max_new_tokens": _pick(
            args.max_new_tokens,
            get("max_new_tokens", "generation"),
        ),
        "temperature": _pick(args.temperature, get("temperature", "generation")),
        "top_p": _pick(args.top_p, get("top_p", "generation")),
        "top_k": _pick(args.top_k, get("top_k", "generation")),
        "min_p": _pick(args.min_p, get("min_p", "generation")),
        "seed": _pick(args.seed, get("seed", "generation")),
        "enable_thinking": _pick(
            getattr(args, "enable_thinking", None),
            get("enable_thinking", "generation"),
        ),
    }

    bootstrap_resamples = int(
        _pick(
            getattr(args, "bootstrap_resamples", None),
            get("bootstrap_resamples", "metrics"),
            1000,
        )
    )
    bootstrap_seed = int(
        _pick(
            getattr(args, "bootstrap_seed", None),
            get("bootstrap_seed", "metrics"),
            42,
        )
    )
    bootstrap_confidence = float(
        _pick(
            getattr(args, "bootstrap_confidence", None),
            get("bootstrap_confidence", "metrics"),
            0.95,
        )
    )
    metric_keys = (
        "num_proc",
        "rm_model_path",
        "cm_model_path",
        "rm_dp_size",
        "rm_tp_size",
        "rm_dtype",
        "rm_trust_remote_code",
        "judge_model",
        "judge_models",
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
        key: _pick(getattr(args, key, None), get(key, "metrics")) for key in metric_keys
    }
    metric_options = {k: v for k, v in metric_options.items() if v is not None}
    cli_judge = getattr(args, "judge_model", None)
    cli_judges = getattr(args, "judge_models", None)
    if cli_judge is not None and cli_judges is not None:
        raise ValueError("--judge-model and --judge-models are mutually exclusive")
    if cli_judges is not None:
        metric_options.pop("judge_model", None)
    elif cli_judge is not None:
        metric_options.pop("judge_models", None)
    if "judge_models" in metric_options:
        if "judge_model" in metric_options:
            raise ValueError("Use either metrics.judge_model or metrics.judge_models")
        from .core.multi_judge import validate_judge_models

        metric_options["judge_models"] = validate_judge_models(metric_options["judge_models"])
    if "num_proc" in metric_options:
        metric_options["num_proc"] = int(metric_options["num_proc"])
        if metric_options["num_proc"] < 1:
            raise ValueError("num_proc must be >= 1")
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
            get("gpu_memory_utilization", "vllm"),
        ),
        "max_model_len": _pick(
            args.max_model_len,
            get("max_model_len", "vllm"),
        ),
        "dtype": _pick(args.dtype, get("dtype", "vllm")),
    }
    cfg_extra_model_kwargs = get("extra_model_kwargs", "vllm")
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
            get("mem_fraction_static", "sglang"),
        ),
        "context_length": _pick(
            getattr(args, "context_length", None),
            get("context_length", "sglang"),
        ),
        "dtype": _pick(args.dtype, get("dtype", "sglang")),
    }
    cfg_sglang_extra = get("extra_model_kwargs", "sglang")
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
    _check_unknown_yaml_keys(cfg, looked_up)

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
