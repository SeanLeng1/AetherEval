from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .io import (
    append_jsonl,
    default_run_id_for_model,
    ensure_dir,
    read_jsonl,
    write_json,
)
from .task_register import BENCHMARKS_DIR, discover_tasks, load_task
from .types import GenerationInput, GenerationRecord, PromptType, Sample
from .vllm_backend import (
    VLLMBackend,
    load_chat_tokenizer,
    render_prompt_with_chat_template,
)


def _info(message: str) -> None:
    print(f"[aethereval] {message}")


def _metric_keys_preview(metrics: dict[str, Any], limit: int = 8) -> str:
    keys = sorted(str(k) for k in metrics.keys())
    if len(keys) <= limit:
        return ", ".join(keys)
    head = ", ".join(keys[:limit])
    return f"{head}, ... (+{len(keys) - limit})"


def _make_progress_bar(total: int, desc: str) -> Any:
    if total <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None
    return tqdm(total=total, desc=desc, unit="gen", dynamic_ncols=True)


def _resolve_primary_metric(
    metrics_module: Any,
    metrics: dict[str, Any],
) -> tuple[str | None, float | None]:
    declared = getattr(metrics_module, "PRIMARY_METRIC", None)
    if declared is not None:
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError("metrics.PRIMARY_METRIC must be a non-empty string when provided.")
        if declared not in metrics:
            raise ValueError(
                f"metrics.PRIMARY_METRIC='{declared}' not found in aggregate output keys: "
                f"{sorted(metrics.keys())}"
            )
        value = metrics.get(declared)
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"metrics.PRIMARY_METRIC='{declared}' must map to numeric value, got {type(value).__name__}."
            )
        return declared, float(value)

    for candidate in ("pass@1", "accuracy", "prompt_level_strict_acc"):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)):
            return candidate, float(value)

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return str(key), float(value)
    return None, None


def _to_sample(item: Any) -> Sample:
    if isinstance(item, Sample):
        return item
    if isinstance(item, dict):
        if "id" not in item:
            raise ValueError("Sample dict must include key 'id'")
        copied = dict(item)
        sample_id = str(copied.pop("id"))
        gold = copied.pop("gold", None)
        meta = copied.pop("meta", {})
        if not isinstance(meta, dict):
            meta = {"value": meta}
        return Sample(id=sample_id, gold=gold, meta=meta, data=copied)
    raise TypeError(f"Unsupported sample type: {type(item).__name__}")


def _to_chat_prompt(prompt: PromptType) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    if isinstance(prompt, list):
        messages: list[dict[str, str]] = []
        for idx, item in enumerate(prompt):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid chat message at index {idx}: expected dict, got {type(item).__name__}"
                )
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", ""))
            messages.append({"role": role, "content": content})
        return messages

    return [{"role": "user", "content": str(prompt)}]


def _parse_tasks_arg(tasks_arg: str, available: list[str]) -> list[str]:
    if tasks_arg.strip() == "all":
        return sorted(available)
    selected = [x.strip() for x in tasks_arg.split(",") if x.strip()]
    if not selected:
        raise ValueError("No tasks selected.")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown tasks: {', '.join(unknown)}. Available: {', '.join(available)}"
        )
    return selected


def _merge_generation_config(
    default_gen: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(default_gen or {})
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg.setdefault("n", 1)
    cfg.setdefault("max_new_tokens", 256)
    cfg.setdefault("temperature", 0.0)
    cfg.setdefault("top_p", 1.0)
    cfg["n"] = int(cfg["n"])
    cfg["max_new_tokens"] = int(cfg["max_new_tokens"])
    cfg["temperature"] = float(cfg["temperature"])
    cfg["top_p"] = float(cfg["top_p"])
    if cfg["n"] < 1:
        raise ValueError(f"n must be >= 1, got {cfg['n']}")
    if cfg["n"] > 1 and cfg["temperature"] == 0.0:
        raise ValueError("n>1 requires temperature>0. Set --temperature > 0.")
    return cfg


def _record_to_json(record: GenerationRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "gen_idx": record.gen_idx,
        "prompt": record.prompt,
        "generation": record.generation,
        "score": record.score,
        "is_pass": record.is_pass,
        "parsed": record.parsed,
        "gold": record.gold,
        "error": record.error,
        "meta": record.meta,
    }


def _load_existing_records(path: Path) -> list[GenerationRecord]:
    rows = read_jsonl(path)
    records: list[GenerationRecord] = []
    for row in rows:
        records.append(
            GenerationRecord(
                sample_id=str(row["sample_id"]),
                gen_idx=int(row["gen_idx"]),
                prompt=row.get("prompt", ""),
                generation=row.get("generation", ""),
                score=float(row.get("score", 0.0)),
                is_pass=bool(row.get("is_pass", False)),
                parsed=row.get("parsed"),
                gold=row.get("gold"),
                error=row.get("error"),
                meta=row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {},
            )
        )
    return records


def _group_records_by_sample(
    records: list[GenerationRecord],
) -> dict[str, list[GenerationRecord]]:
    grouped: dict[str, list[GenerationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sample_id].append(record)
    for sample_id in grouped:
        grouped[sample_id].sort(key=lambda x: x.gen_idx)
    return grouped


def _build_sample_results(
    samples: list[Sample],
    grouped_records: dict[str, list[GenerationRecord]],
) -> list[dict[str, Any]]:
    sample_results: list[dict[str, Any]] = []
    for sample in samples:
        records = grouped_records.get(sample.id, [])
        sample_results.append(
            {
                "sample_id": sample.id,
                "gold": sample.gold,
                "meta": sample.meta,
                "scores": [r.score for r in records],
                "passes": [r.is_pass for r in records],
                "records": [_record_to_json(r) for r in records],
            }
        )
    return sample_results


def _call_task_aggregate(
    aggregate_fn: Any,
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any],
) -> dict[str, Any]:
    result = aggregate_fn(sample_results, metric_options)

    if not isinstance(result, dict):
        raise ValueError("aggregate must return a dict[str, float]")
    return result


def _aggregate_run_metrics(task_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for summary in task_summaries.values():
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                grouped[key].append(float(value))

    aggregate_metrics: dict[str, float] = {}
    for key, values in grouped.items():
        if values:
            aggregate_metrics[key] = sum(values) / len(values)

    return {
        "num_tasks": len(task_summaries),
        "metrics": aggregate_metrics,
    }


def _run_single_task(
    *,
    task_name: str,
    task_module: Any,
    metrics_module: Any,
    task_dir: Path,
    backend: VLLMBackend,
    task_output_dir: Path,
    gen_overrides: dict[str, Any],
    metric_options: dict[str, Any],
    overwrite: bool,
    run_config_common: dict[str, Any],
) -> dict[str, Any]:
    _info(f"[{task_name}] loading task from {task_dir}")
    samples_raw = task_module.load_samples(task_dir)
    samples = [_to_sample(item) for item in samples_raw]
    sample_id_set = set()
    for sample in samples:
        if sample.id in sample_id_set:
            raise ValueError(f"Duplicate sample id in task '{task_name}': {sample.id}")
        sample_id_set.add(sample.id)

    gen_cfg = _merge_generation_config(task_module.DEFAULT_GEN, gen_overrides)
    n = int(gen_cfg["n"])
    _info(
        f"[{task_name}] samples={len(samples)} n={n} overwrite={overwrite} "
        f"data_file={getattr(task_module, 'DATA_FILE', '(unknown)')}"
    )

    ensure_dir(task_output_dir)
    predictions_path = task_output_dir / "predictions.jsonl"
    summary_path = task_output_dir / "summary.json"
    run_config_path = task_output_dir / "run_config.json"

    if overwrite and predictions_path.exists():
        _info(f"[{task_name}] overwrite enabled: removing {predictions_path}")
        predictions_path.unlink()

    existing_records: list[GenerationRecord] = []
    if predictions_path.exists():
        _info(f"[{task_name}] resume: loading existing predictions from {predictions_path}")
        raw_existing = _load_existing_records(predictions_path)
        dedup: dict[tuple[str, int], GenerationRecord] = {}
        for record in raw_existing:
            if record.sample_id not in sample_id_set:
                continue
            if record.gen_idx < 0 or record.gen_idx >= n:
                continue
            dedup[(record.sample_id, record.gen_idx)] = record
        existing_records = list(dedup.values())

    existing_lookup: dict[str, set[int]] = defaultdict(set)
    for record in existing_records:
        existing_lookup[record.sample_id].add(record.gen_idx)

    pending_inputs: list[GenerationInput] = []
    pending_indices: dict[str, list[int]] = {}
    pending_record_count = 0
    for sample in samples:
        missing = [i for i in range(n) if i not in existing_lookup.get(sample.id, set())]
        pending_indices[sample.id] = missing
        pending_record_count += len(missing)
        if not missing:
            continue
        prompt = _to_chat_prompt(task_module.build_prompt(sample))
        pending_inputs.append(
            GenerationInput(
                sample_id=sample.id,
                prompt=prompt,
                num_generations=len(missing),
            )
        )
    _info(
        f"[{task_name}] existing_records={len(existing_records)} pending_samples={len(pending_inputs)} "
        f"pending_records={pending_record_count}"
    )

    samples_by_id = {sample.id: sample for sample in samples}
    new_records: list[GenerationRecord] = []

    if pending_inputs:
        _info(f"[{task_name}] starting vLLM generation")
        generated_outputs = backend.generate(pending_inputs, gen_cfg)
        score_bar = _make_progress_bar(pending_record_count, f"[{task_name}] scoring")
        try:
            for output in generated_outputs:
                sample = samples_by_id[output.sample_id]
                missing = pending_indices[output.sample_id]
                generations = list(output.generations)
                if len(generations) < len(missing):
                    generations.extend([""] * (len(missing) - len(generations)))

                rows_to_write: list[dict[str, Any]] = []
                for local_idx, gen_idx in enumerate(missing):
                    generation_text = generations[local_idx]
                    error = output.error
                    score = 0.0
                    parsed = None
                    meta: dict[str, Any] = {}
                    is_pass = False

                    if error is None:
                        try:
                            scored = metrics_module.score_generation(sample, generation_text)
                            if "score" not in scored:
                                raise ValueError("score_generation must return key 'score'.")
                            score = float(scored["score"])
                            parsed = scored.get("parsed")
                            task_meta = scored.get("meta", {})
                            meta = task_meta if isinstance(task_meta, dict) else {"value": task_meta}
                            is_pass = bool(scored.get("is_pass", False))
                        except Exception as exc:  # noqa: BLE001
                            error = f"score_generation error: {type(exc).__name__}: {exc}"

                    record = GenerationRecord(
                        sample_id=sample.id,
                        gen_idx=gen_idx,
                        prompt=output.prompt,
                        generation=generation_text,
                        score=score,
                        is_pass=is_pass,
                        parsed=parsed,
                        gold=sample.gold,
                        error=error,
                        meta=meta,
                    )
                    new_records.append(record)
                    rows_to_write.append(_record_to_json(record))
                    if score_bar is not None:
                        score_bar.update(1)

                append_jsonl(predictions_path, rows_to_write)
        finally:
            if score_bar is not None:
                score_bar.close()
        _info(f"[{task_name}] generation finished: new_records={len(new_records)}")
    else:
        _info(f"[{task_name}] no pending generations; skip inference")

    all_records = existing_records + new_records
    grouped_records = _group_records_by_sample(all_records)
    sample_results = _build_sample_results(samples, grouped_records)

    aggregate_result = _call_task_aggregate(
        metrics_module.aggregate,
        sample_results,
        {**metric_options, "n": n},
    )
    warnings = aggregate_result.pop("__warnings__", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)]

    metrics = aggregate_result
    primary_metric, primary_score = _resolve_primary_metric(metrics_module, metrics)

    summary = {
        "task": task_name,
        "num_samples": len(samples),
        "n": n,
        "existing_records": len(existing_records),
        "new_records": len(new_records),
        "total_records": len(all_records),
        "metrics": metrics,
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "warnings": warnings,
    }
    _info(
        f"[{task_name}] aggregate done: total_records={len(all_records)} "
        f"metrics=[{_metric_keys_preview(metrics)}]"
    )
    if warnings:
        _info(f"[{task_name}] warnings={warnings}")

    task_run_config = dict(run_config_common)
    task_run_config.update(
        {
            "task": task_name,
            "task_dir": str(task_dir),
            "generation_config": gen_cfg,
            "metric_options": {**metric_options, "n": n},
            "overwrite": overwrite,
        }
    )

    write_json(summary_path, summary)
    write_json(run_config_path, task_run_config)
    return summary


def run_evaluation(
    *,
    model: str,
    tasks: str,
    output_dir: str | Path,
    dp_size: int = 1,
    tensor_parallel_size: int = 1,
    gen_overrides: dict[str, Any] | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    bootstrap_confidence: float = 0.95,
    overwrite: bool = False,
    run_id: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    backend: VLLMBackend | None = None,
    benchmarks_dir: Path | None = None,
) -> dict[str, Any]:
    task_root = benchmarks_dir or BENCHMARKS_DIR
    tasks_map = discover_tasks(task_root)
    available = sorted(tasks_map.keys())
    if not available:
        raise RuntimeError(f"No tasks found in {task_root}")

    selected = _parse_tasks_arg(tasks, available)
    out_dir = Path(output_dir)
    this_run_id = run_id or default_run_id_for_model(model)
    run_root = out_dir / this_run_id
    ensure_dir(run_root)
    _info(f"benchmark_root={task_root}")
    _info(f"discovered_tasks={len(available)} selected={selected}")
    _info(
        f"model={model} dp_size={int(dp_size)} tp_size={int(tensor_parallel_size)} "
        f"output_dir={out_dir} run_id={this_run_id}"
    )
    if model_kwargs:
        _info(f"vllm_model_kwargs={model_kwargs}")
    _info(f"run_output_dir={run_root}")

    created_backend = False
    if backend is None:
        backend = VLLMBackend(
            model=model,
            dp_size=dp_size,
            tensor_parallel_size=tensor_parallel_size,
            model_kwargs=model_kwargs,
        )
        created_backend = True

    try:
        run_config_common = {
            "model": model,
            "dp_size": int(dp_size),
            "tp_size": int(tensor_parallel_size),
            "model_kwargs": model_kwargs or {},
        }
        metric_options = {
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_confidence": float(bootstrap_confidence),
        }
        summaries: dict[str, Any] = {}
        for task_name in selected:
            _info(f"===== start task: {task_name} =====")
            bundle = load_task(task_name, task_root)
            task_spec = tasks_map[task_name]
            task_output_dir = run_root / task_name
            summary = _run_single_task(
                task_name=task_name,
                task_module=bundle.task_module,
                metrics_module=bundle.metrics_module,
                task_dir=task_spec.task_dir,
                backend=backend,
                task_output_dir=task_output_dir,
                gen_overrides=gen_overrides or {},
                metric_options=metric_options,
                overwrite=overwrite,
                run_config_common=run_config_common,
            )
            summaries[task_name] = summary
            _info(f"===== finish task: {task_name} =====")

        primary_scores: dict[str, dict[str, Any]] = {}
        for task_name, task_summary in summaries.items():
            primary_scores[task_name] = {
                "metric": task_summary.get("primary_metric"),
                "score": task_summary.get("primary_score"),
            }

        run_summary = {
            "run_id": this_run_id,
            "tasks": selected,
            "model": model,
            "results": summaries,
            "primary_scores": primary_scores,
            "summary": _aggregate_run_metrics(summaries),
        }
        write_json(run_root / "run_summary.json", run_summary)
        _info(f"run_summary_path={run_root / 'run_summary.json'}")
        return run_summary
    finally:
        if created_backend:
            backend.close()


def inspect_prompts(
    *,
    model: str,
    tasks: str,
    model_kwargs: dict[str, Any] | None = None,
    benchmarks_dir: Path | None = None,
    inspect_limit: int = 5,
    prompt_renderer: Callable[[PromptType], str] | None = None,
) -> dict[str, Any]:
    task_root = benchmarks_dir or BENCHMARKS_DIR
    tasks_map = discover_tasks(task_root)
    available = sorted(tasks_map.keys())
    if not available:
        raise RuntimeError(f"No tasks found in {task_root}")

    selected = _parse_tasks_arg(tasks, available)
    limit = max(1, int(inspect_limit))
    _info(f"inspect mode: model={model} tasks={selected} limit={limit}")

    if prompt_renderer is None:
        tokenizer = load_chat_tokenizer(model, model_kwargs)
        prompt_renderer = lambda prompt: render_prompt_with_chat_template(prompt, tokenizer)

    task_results: dict[str, list[dict[str, Any]]] = {}
    for task_name in selected:
        bundle = load_task(task_name, task_root)
        task_spec = tasks_map[task_name]
        samples_raw = bundle.task_module.load_samples(task_spec.task_dir)
        samples = [_to_sample(item) for item in samples_raw]

        rows: list[dict[str, Any]] = []
        for sample in samples[:limit]:
            prompt = _to_chat_prompt(bundle.task_module.build_prompt(sample))
            rendered = str(prompt_renderer(prompt))
            rows.append(
                {
                    "sample_id": sample.id,
                    "prompt": rendered,
                }
            )
        task_results[task_name] = rows
        _info(f"[inspect:{task_name}] samples={len(samples)} shown={len(rows)}")

    return {
        "model": model,
        "tasks": selected,
        "inspect_limit": limit,
        "results": task_results,
    }
