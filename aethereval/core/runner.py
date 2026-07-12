import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .io import (
    append_jsonl,
    default_run_id_for_model,
    ensure_dir,
    read_jsonl,
    run_output_dir,
    write_json,
)
from .task_register import BENCHMARKS_DIR, discover_tasks, load_task
from .types import (
    GenerationInput,
    GenerationOutput,
    GenerationRecord,
    PromptType,
    Sample,
)
from aethereval.backends import (
    GenerationBackend,
    count_text_tokens,
    create_backend,
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
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit="gen", dynamic_ncols=True)


def _resolve_primary_metric(
    metrics_module: Any,
    metrics: dict[str, Any],
) -> tuple[str | None, float | None]:
    declared = getattr(metrics_module, "PRIMARY_METRIC", None)
    if declared is not None:
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(
                "metrics.PRIMARY_METRIC must be a non-empty string when provided."
            )
        if declared not in metrics:
            raise ValueError(
                f"metrics.PRIMARY_METRIC='{declared}' not found in aggregate output keys: "
                f"{sorted(metrics.keys())}"
            )
        value = metrics[declared]
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
            raise ValueError(f"Sample '{sample_id}' meta must be a dict")
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
            role = str(item["role"]).strip()
            content = str(item["content"])
            if not role:
                raise ValueError(f"Invalid chat message at index {idx}: empty role")
            messages.append({"role": role, "content": content})
        return messages

    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


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
    cfg.setdefault("top_k", -1)
    cfg["n"] = int(cfg["n"])
    cfg["max_new_tokens"] = int(cfg["max_new_tokens"])
    cfg["temperature"] = float(cfg["temperature"])
    cfg["top_p"] = float(cfg["top_p"])
    cfg["top_k"] = int(cfg["top_k"]) if cfg.get("top_k") is not None else -1
    if cfg["n"] < 1:
        raise ValueError(f"n must be >= 1, got {cfg['n']}")
    if cfg["n"] > 1 and cfg["temperature"] == 0.0:
        raise ValueError("n>1 requires temperature>0. Set --temperature > 0.")
    return cfg


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _tokenizer_getter(
    *,
    backend: GenerationBackend,
    model: str,
    model_kwargs: dict[str, Any] | None,
) -> Callable[[], Any]:
    cache = {"tokenizer": getattr(backend, "_tokenizer", None)}

    def _get() -> Any:
        tokenizer = cache["tokenizer"] or getattr(backend, "_tokenizer", None)
        if tokenizer is None:
            tokenizer = load_chat_tokenizer(model, model_kwargs)
        cache["tokenizer"] = tokenizer
        return tokenizer

    return _get


def _token_count_from_text(text: str, tokenizer_getter: Callable[[], Any]) -> int:
    return count_text_tokens(text, tokenizer_getter())


def _normalize_response_token_counts(
    *,
    output: GenerationOutput,
    tokenizer_getter: Callable[[], Any],
) -> list[int]:
    raw_counts = output.meta.get("response_token_counts")
    if raw_counts is None:
        counts: list[int | None] = [None for _ in output.generations]
    elif isinstance(raw_counts, list) and len(raw_counts) == len(output.generations):
        counts = []
        for idx, value in enumerate(raw_counts):
            if value is None:
                counts.append(None)
            elif _is_token_count(value):
                counts.append(int(value))
            else:
                raise ValueError(
                    f"Invalid response_token_counts[{idx}] for sample {output.sample_id}: {value!r}"
                )
    else:
        raise ValueError(
            "GenerationOutput.meta['response_token_counts'] must be a list aligned "
            f"with generations for sample {output.sample_id}."
        )

    normalized: list[int] = []
    for generation, count in zip(output.generations, counts, strict=True):
        if count is None:
            count = _token_count_from_text(generation, tokenizer_getter)
        normalized.append(count)
    return normalized


def _ensure_output_token_metadata(
    *,
    output: GenerationOutput,
    tokenizer_getter: Callable[[], Any],
) -> None:
    if not isinstance(output.meta, dict):
        raise ValueError(f"GenerationOutput meta must be a dict for {output.sample_id}")

    prompt_count = output.meta.get("prompt_token_count")
    if prompt_count is None:
        rendered_prompt = render_prompt_with_chat_template(
            output.prompt, tokenizer_getter()
        )
        prompt_count = _token_count_from_text(rendered_prompt, tokenizer_getter)
    elif not _is_token_count(prompt_count):
        raise ValueError(
            f"Invalid prompt_token_count for sample {output.sample_id}: {prompt_count!r}"
        )

    output.meta["prompt_token_count"] = int(prompt_count)
    output.meta["response_token_counts"] = _normalize_response_token_counts(
        output=output,
        tokenizer_getter=tokenizer_getter,
    )


def _ensure_outputs_token_metadata(
    outputs: list[GenerationOutput],
    tokenizer_getter: Callable[[], Any],
) -> None:
    for output in outputs:
        _ensure_output_token_metadata(output=output, tokenizer_getter=tokenizer_getter)


def _generation_token_meta(output: GenerationOutput, local_idx: int) -> dict[str, int]:
    prompt_count = output.meta["prompt_token_count"]
    response_counts = output.meta["response_token_counts"]
    if not _is_token_count(prompt_count):
        raise ValueError(f"Invalid prompt_token_count for sample {output.sample_id}")
    if (
        not isinstance(response_counts, list)
        or local_idx >= len(response_counts)
        or not _is_token_count(response_counts[local_idx])
    ):
        raise ValueError(
            f"Invalid response_token_counts for sample {output.sample_id}"
        )
    return {
        "prompt_token_count": int(prompt_count),
        "response_token_count": int(response_counts[local_idx]),
    }


def _token_usage_summary(records: list[GenerationRecord]) -> dict[str, Any]:
    prompt_counts: list[int] = []
    response_counts: list[int] = []
    for record in records:
        prompt_count = record.meta.get("prompt_token_count")
        response_count = record.meta.get("response_token_count")
        if not _is_token_count(prompt_count) or not _is_token_count(response_count):
            raise ValueError(
                f"Missing token counts in record meta for sample {record.sample_id}"
            )
        prompt_counts.append(int(prompt_count))
        response_counts.append(int(response_count))

    if not records:
        return {
            "avg_prompt_tokens": None,
            "avg_response_tokens": None,
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
        }

    return {
        "avg_prompt_tokens": sum(prompt_counts) / len(prompt_counts),
        "avg_response_tokens": sum(response_counts) / len(response_counts),
        "total_prompt_tokens": sum(prompt_counts),
        "total_response_tokens": sum(response_counts),
    }


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
        meta = row["meta"]
        if not isinstance(meta, dict):
            raise ValueError(f"Existing prediction meta must be a dict in {path}")
        records.append(
            GenerationRecord(
                sample_id=str(row["sample_id"]),
                gen_idx=int(row["gen_idx"]),
                prompt=row["prompt"],
                generation=row["generation"],
                score=float(row["score"]),
                is_pass=bool(row["is_pass"]),
                parsed=row["parsed"] if "parsed" in row else None,
                gold=row["gold"] if "gold" in row else None,
                error=row["error"] if "error" in row else None,
                meta=meta,
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


def _score_generation(
    *,
    metrics_module: Any,
    sample: Sample,
    generation: str,
) -> tuple[float, bool, Any, dict[str, Any]]:
    scored = metrics_module.score_generation(sample, generation)
    return _normalize_score_generation_result(scored)


def _normalize_score_generation_result(
    scored: Any,
) -> tuple[float, bool, Any, dict[str, Any]]:
    if not isinstance(scored, dict):
        raise TypeError("score_generation result must be a dict.")
    if "score" not in scored:
        raise ValueError("score_generation result must include key 'score'.")

    score = float(scored["score"])
    parsed = scored.get("parsed")
    task_meta = scored.get("meta", {})
    if not isinstance(task_meta, dict):
        raise TypeError("score_generation 'meta' must be a dict when provided.")
    is_pass = bool(scored["is_pass"]) if "is_pass" in scored else score >= 1.0
    return score, is_pass, parsed, task_meta


def _normalize_batch_score_results(
    *,
    raw_results: Any,
    outputs: list[GenerationOutput],
) -> dict[str, list[tuple[float, bool, Any, dict[str, Any]]]]:
    if not isinstance(raw_results, list):
        raise TypeError("score_generations_batch must return list[list[dict]].")
    if len(raw_results) != len(outputs):
        raise ValueError(
            "score_generations_batch output length mismatch: "
            f"got {len(raw_results)}, expected {len(outputs)}"
        )

    normalized: dict[str, list[tuple[float, bool, Any, dict[str, Any]]]] = {}
    for output, per_generation in zip(outputs, raw_results, strict=True):
        if output.sample_id in normalized:
            raise ValueError(
                f"score_generations_batch received duplicate sample id: {output.sample_id}"
            )
        if not isinstance(per_generation, list):
            raise TypeError(
                "score_generations_batch must return a list of score dicts for "
                f"sample {output.sample_id}"
            )
        if len(per_generation) != len(output.generations):
            raise ValueError(
                "score_generations_batch per-sample length mismatch for "
                f"sample {output.sample_id}: got {len(per_generation)}, "
                f"expected {len(output.generations)}"
            )
        normalized[output.sample_id] = [
            _normalize_score_generation_result(item) for item in per_generation
        ]
    return normalized


def _score_generation_outputs(
    *,
    metrics_module: Any,
    samples_by_id: dict[str, Sample],
    outputs: list[GenerationOutput],
    metric_options: dict[str, Any],
    runtime_metric_options: dict[str, Any] | None = None,
    total_records: int,
    progress_desc: str,
) -> dict[str, list[tuple[float, bool, Any, dict[str, Any]]]]:
    batch_score_fn = getattr(metrics_module, "score_generations_batch", None)
    if callable(batch_score_fn):
        samples = [samples_by_id[output.sample_id] for output in outputs]
        score_options = dict(metric_options)
        if runtime_metric_options:
            score_options.update(runtime_metric_options)
        raw_results = batch_score_fn(samples, outputs, score_options)
        return _normalize_batch_score_results(raw_results=raw_results, outputs=outputs)

    score_bar = _make_progress_bar(total_records, progress_desc)
    scored: dict[str, list[tuple[float, bool, Any, dict[str, Any]]]] = {}
    try:
        for output in outputs:
            sample = samples_by_id[output.sample_id]
            scored[output.sample_id] = []
            for generation_text in output.generations:
                scored[output.sample_id].append(
                    _score_generation(
                        metrics_module=metrics_module,
                        sample=sample,
                        generation=generation_text,
                    )
                )
                if score_bar is not None:
                    score_bar.update(1)
    finally:
        if score_bar is not None:
            score_bar.close()
    return scored


def _records_to_generation_outputs(
    records: list[GenerationRecord],
) -> tuple[list[GenerationOutput], dict[str, list[int]]]:
    grouped = _group_records_by_sample(records)
    outputs: list[GenerationOutput] = []
    gen_indices: dict[str, list[int]] = {}
    for sample_id, sample_records in grouped.items():
        prompt_counts = [
            record.meta.get("prompt_token_count")
            for record in sample_records
            if record.meta.get("prompt_token_count") is not None
        ]
        meta: dict[str, Any] = {}
        if prompt_counts:
            meta["prompt_token_count"] = prompt_counts[0]
        response_counts: list[int | None] = [
            record.meta.get("response_token_count") for record in sample_records
        ]
        if any(count is not None for count in response_counts):
            meta["response_token_counts"] = response_counts
        outputs.append(
            GenerationOutput(
                sample_id=sample_id,
                prompt=sample_records[0].prompt,
                generations=[record.generation for record in sample_records],
                meta=meta,
            )
        )
        gen_indices[sample_id] = [record.gen_idx for record in sample_records]
    return outputs, gen_indices


def _aggregate_run_metrics(task_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for summary in task_summaries.values():
        metrics = summary["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError("Task summary metrics must be a dict")
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


def _aggregate_primary_scores(
    task_summaries: dict[str, dict[str, Any]],
) -> float | None:
    values: list[float] = []
    for summary in task_summaries.values():
        value = summary["primary_score"]
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _load_existing_task_summaries(
    *,
    run_root: Path,
    available_tasks: set[str],
    skip_tasks: set[str],
) -> dict[str, dict[str, Any]]:
    if not run_root.exists():
        return {}

    loaded: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(run_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if task_name in skip_tasks:
            continue
        if task_name not in available_tasks:
            continue
        summary_path = task_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Existing summary must be a JSON object: {summary_path}")
        loaded[task_name] = data
    return loaded


def _run_single_task(
    *,
    task_name: str,
    task_module: Any,
    metrics_module: Any,
    task_dir: Path,
    backend: GenerationBackend,
    task_output_dir: Path,
    gen_overrides: dict[str, Any],
    metric_options: dict[str, Any],
    overwrite: bool,
    run_config_common: dict[str, Any],
    tokenizer_getter: Callable[[], Any],
) -> dict[str, Any]:
    _info(f"[{task_name}] loading task from {task_dir}")
    samples_raw = task_module.load_samples(task_dir)
    samples = [_to_sample(item) for item in samples_raw]
    samples_by_id = {sample.id: sample for sample in samples}
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
        _info(
            f"[{task_name}] resume: loading existing predictions from {predictions_path}"
        )
        raw_existing = _load_existing_records(predictions_path)
        dedup: dict[tuple[str, int], GenerationRecord] = {}
        for record in raw_existing:
            if record.sample_id not in sample_id_set:
                raise ValueError(
                    f"[{task_name}] existing prediction references unknown sample id: {record.sample_id}"
                )
            if record.gen_idx < 0 or record.gen_idx >= n:
                raise ValueError(
                    f"[{task_name}] existing prediction gen_idx out of range for n={n}: "
                    f"sample_id={record.sample_id} gen_idx={record.gen_idx}"
                )
            if record.error is not None:
                raise ValueError(
                    f"[{task_name}] existing prediction contains backend error for "
                    f"sample_id={record.sample_id} gen_idx={record.gen_idx}: {record.error}"
                )
            dedup[(record.sample_id, record.gen_idx)] = record
        existing_records = list(dedup.values())

    if existing_records:
        _info(
            f"[{task_name}] resume: rescoring existing records ({len(existing_records)})"
        )
        rescored_existing: list[GenerationRecord] = []
        existing_outputs, existing_gen_indices = _records_to_generation_outputs(
            existing_records
        )
        _ensure_outputs_token_metadata(existing_outputs, tokenizer_getter)
        runtime_metric_options = {}
        if getattr(metrics_module, "REQUIRES_TOKENIZER", False):
            runtime_metric_options["_tokenizer"] = tokenizer_getter()
        if getattr(metrics_module, "REQUIRES_BACKEND", False):
            runtime_metric_options["_backend"] = backend
        existing_scores = _score_generation_outputs(
            metrics_module=metrics_module,
            samples_by_id=samples_by_id,
            outputs=existing_outputs,
            metric_options={**metric_options, "n": n},
            runtime_metric_options=runtime_metric_options,
            total_records=len(existing_records),
            progress_desc=f"[{task_name}] rescoring",
        )
        existing_records_by_id = _group_records_by_sample(existing_records)
        for output in existing_outputs:
            sample = samples_by_id[output.sample_id]
            original_records = existing_records_by_id[output.sample_id]
            scores = existing_scores[output.sample_id]
            gen_indices = existing_gen_indices[output.sample_id]
            for local_idx, (record, gen_idx, scored) in enumerate(
                zip(original_records, gen_indices, scores, strict=True)
            ):
                score, is_pass, parsed, meta = scored
                record_meta = dict(meta)
                record_meta.update(_generation_token_meta(output, local_idx))
                rescored_existing.append(
                    GenerationRecord(
                        sample_id=record.sample_id,
                        gen_idx=gen_idx,
                        prompt=record.prompt,
                        generation=record.generation,
                        score=score,
                        is_pass=is_pass,
                        parsed=parsed,
                        gold=sample.gold,
                        error=record.error,
                        meta=record_meta,
                    )
                )

        existing_records = rescored_existing
        _info(f"[{task_name}] resume rescoring finished")
        if predictions_path.exists():
            predictions_path.unlink()
        append_jsonl(
            predictions_path, (_record_to_json(record) for record in existing_records)
        )
    elif predictions_path.exists():
        # If resume file has no reusable rows under current settings, rebuild it from pending outputs only.
        predictions_path.unlink()

    existing_lookup: dict[str, set[int]] = defaultdict(set)
    for record in existing_records:
        existing_lookup[record.sample_id].add(record.gen_idx)

    pending_inputs: list[GenerationInput] = []
    pending_indices: dict[str, list[int]] = {}
    pending_record_count = 0
    for sample in samples:
        missing = [
            i for i in range(n) if i not in existing_lookup.get(sample.id, set())
        ]
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

    new_records: list[GenerationRecord] = []

    if pending_inputs:
        backend_label = getattr(backend, "name", "backend")
        _info(f"[{task_name}] starting {backend_label} generation")
        generated_outputs = backend.generate(pending_inputs, gen_cfg)
        outputs_by_sample: dict[str, Any] = {}
        for output in generated_outputs:
            if output.sample_id in outputs_by_sample:
                raise ValueError(
                    f"[{task_name}] backend returned duplicate output for sample {output.sample_id}"
                )
            outputs_by_sample[output.sample_id] = output
        expected_output_ids = {item.sample_id for item in pending_inputs}
        returned_output_ids = set(outputs_by_sample.keys())
        if returned_output_ids != expected_output_ids:
            missing_ids = sorted(expected_output_ids - returned_output_ids)
            extra_ids = sorted(returned_output_ids - expected_output_ids)
            raise ValueError(
                f"[{task_name}] backend output sample ids mismatch; "
                f"missing={missing_ids} extra={extra_ids}"
            )
        for output in generated_outputs:
            missing = pending_indices[output.sample_id]
            generations = list(output.generations)
            if output.error is not None:
                raise RuntimeError(
                    f"[{task_name}] backend failed for sample {output.sample_id}: {output.error}"
                )
            if len(generations) != len(missing):
                raise ValueError(
                    f"[{task_name}] backend returned {len(generations)} generations for "
                    f"sample {output.sample_id}, expected {len(missing)}"
                )

        _ensure_outputs_token_metadata(generated_outputs, tokenizer_getter)
        runtime_metric_options = {}
        if getattr(metrics_module, "REQUIRES_TOKENIZER", False):
            runtime_metric_options["_tokenizer"] = tokenizer_getter()
        if getattr(metrics_module, "REQUIRES_BACKEND", False):
            runtime_metric_options["_backend"] = backend

        generated_scores = _score_generation_outputs(
            metrics_module=metrics_module,
            samples_by_id=samples_by_id,
            outputs=generated_outputs,
            metric_options={**metric_options, "n": n},
            runtime_metric_options=runtime_metric_options,
            total_records=pending_record_count,
            progress_desc=f"[{task_name}] scoring",
        )

        rows_to_write: list[dict[str, Any]] = []
        for output in generated_outputs:
            sample = samples_by_id[output.sample_id]
            missing = pending_indices[output.sample_id]
            generations = list(output.generations)
            scores = generated_scores[output.sample_id]
            for local_idx, gen_idx in enumerate(missing):
                score, is_pass, parsed, meta = scores[local_idx]
                record_meta = dict(meta)
                record_meta.update(_generation_token_meta(output, local_idx))
                record = GenerationRecord(
                    sample_id=sample.id,
                    gen_idx=gen_idx,
                    prompt=output.prompt,
                    generation=generations[local_idx],
                    score=score,
                    is_pass=is_pass,
                    parsed=parsed,
                    gold=sample.gold,
                    error=None,
                    meta=record_meta,
                )
                new_records.append(record)
                rows_to_write.append(_record_to_json(record))

        append_jsonl(predictions_path, rows_to_write)
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
    token_usage = _token_usage_summary(all_records)
    if token_usage["avg_prompt_tokens"] is not None:
        metrics["avg_prompt_tokens"] = token_usage["avg_prompt_tokens"]
    if token_usage["avg_response_tokens"] is not None:
        metrics["avg_response_tokens"] = token_usage["avg_response_tokens"]
    primary_metric, primary_score = _resolve_primary_metric(metrics_module, metrics)

    summary = {
        "task": task_name,
        "num_samples": len(samples),
        "n": n,
        "existing_records": len(existing_records),
        "new_records": len(new_records),
        "total_records": len(all_records),
        "metrics": metrics,
        "token_usage": token_usage,
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
    metric_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    run_id: str | None = None,
    backend_name: str = "vllm",
    backend_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    backend: GenerationBackend | None = None,
    benchmarks_dir: Path | None = None,
) -> dict[str, Any]:
    effective_model_kwargs = (
        backend_kwargs if backend_kwargs is not None else model_kwargs
    )
    task_root = benchmarks_dir or BENCHMARKS_DIR
    tasks_map = discover_tasks(task_root)
    available = sorted(tasks_map.keys())
    if not available:
        raise RuntimeError(f"No tasks found in {task_root}")

    selected = _parse_tasks_arg(tasks, available)
    out_dir = Path(output_dir)
    this_run_id = run_id or default_run_id_for_model(model)
    run_root = run_output_dir(out_dir, model, run_id)
    ensure_dir(run_root)
    _info(f"benchmark_root={task_root}")
    _info(f"discovered_tasks={len(available)} selected={selected}")
    _info(
        f"model={model} backend={backend_name} dp_size={int(dp_size)} "
        f"tp_size={int(tensor_parallel_size)} output_dir={out_dir} run_id={this_run_id}"
    )
    if effective_model_kwargs:
        _info(f"backend_model_kwargs={effective_model_kwargs}")
    _info(f"run_output_dir={run_root}")

    created_backend = False
    if backend is None:
        backend = create_backend(
            backend_name=backend_name,
            model=model,
            dp_size=dp_size,
            tensor_parallel_size=tensor_parallel_size,
            model_kwargs=effective_model_kwargs,
        )
        created_backend = True
    backend_label = getattr(backend, "name", backend_name)
    get_tokenizer = _tokenizer_getter(
        backend=backend,
        model=model,
        model_kwargs=effective_model_kwargs,
    )

    try:
        run_config_common = {
            "model": model,
            "backend": backend_label,
            "dp_size": int(dp_size),
            "tp_size": int(tensor_parallel_size),
            "model_kwargs": effective_model_kwargs or {},
        }
        resolved_metric_options = {
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_confidence": float(bootstrap_confidence),
        }
        if metric_options:
            resolved_metric_options.update(metric_options)
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
                metric_options=resolved_metric_options,
                overwrite=overwrite,
                run_config_common=run_config_common,
                tokenizer_getter=get_tokenizer,
            )
            summaries[task_name] = summary
            _info(f"===== finish task: {task_name} =====")

        existing_summaries = _load_existing_task_summaries(
            run_root=run_root,
            available_tasks=set(available),
            skip_tasks=set(selected),
        )
        if existing_summaries:
            _info(
                "including existing task summaries in run-level aggregation: "
                f"{sorted(existing_summaries.keys())}"
            )

        all_task_summaries = {**existing_summaries, **summaries}
        all_primary_scores = {
            task_name: {
                "metric": task_summary["primary_metric"],
                "score": task_summary["primary_score"],
            }
            for task_name, task_summary in all_task_summaries.items()
        }

        run_summary = {
            "run_id": this_run_id,
            "selected_tasks": selected,
            "tasks": sorted(all_task_summaries.keys()),
            "model": model,
            "backend": backend_label,
            "results": all_task_summaries,
            "primary_scores": all_primary_scores,
            "primary_score_aggregate": _aggregate_primary_scores(all_task_summaries),
            "summary": _aggregate_run_metrics(all_task_summaries),
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

        def _render(prompt: PromptType) -> str:
            return render_prompt_with_chat_template(prompt, tokenizer)

        prompt_renderer = _render

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
