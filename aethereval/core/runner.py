import dataclasses
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from aethereval.metrics.common import strip_reasoning
from aethereval.progress import Progress

from .io import (
    append_jsonl,
    ensure_dir,
    model_output_name,
    read_jsonl,
    run_output_dir,
    write_json,
    write_jsonl,
)
from .run_summary import build_run_summary, load_task_summaries, phase_name
from .primary_score import primary_score_fields, primary_score_scale
from .task_register import (
    BENCHMARKS_DIR,
    _load_module_from_path,
    discover_tasks,
    load_task,
    parse_task_names,
)
from .task_defaults import (
    resolve_phase_num_repeats,
    resolve_task_default_gen,
    resolve_task_default_metrics,
)
from .types import (
    GenerationInput,
    GenerationOutput,
    GenerationRecord,
    PromptType,
    Sample,
)
from aethereval.backends import (
    GenerationBackend,
    chat_template_kwargs_from_generation_config,
    count_text_tokens,
    create_backend,
    load_chat_tokenizer,
    normalize_backend_name,
    render_prompt_with_chat_template,
)
from benchmark_utils.local_judge import OfflineJudgeClient

_UNSCORED_META_KEY = "_aethereval_unscored"
_JUDGE_META_KEY = "_aethereval_judge"
# Judge options that change only throughput or credentials, not the judgments.
_JUDGE_TRANSPORT_OPTIONS = {
    "judge_workers",
    "judge_timeout",
    "judge_max_retries",
    "judge_api_key_env",
    "judge_dp_size",
    "judge_tp_size",
}
# Scorer and backend packages whose versions can move scores; recorded per eval.
_SCORING_PACKAGES = (
    "math-verify",
    "latex2sympy2_extended",
    "evalplus",
    "nltk",
    "langdetect",
    "sglang",
    "vllm",
)
_SCORE_WORKER_METRICS: Any = None


def _info(message: str) -> None:
    print(f"[aethereval] {message}")


def _metric_keys_preview(metrics: dict[str, Any], limit: int = 8) -> str:
    keys = sorted(str(k) for k in metrics.keys())
    if len(keys) <= limit:
        return ", ".join(keys)
    head = ", ".join(keys[:limit])
    return f"{head}, ... (+{len(keys) - limit})"


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


def _merge_generation_config(
    default_gen: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    cfg = dict(default_gen or {})
    default_n = int(cfg.get("n", 1))
    use_task_sampling_temperature = (
        overrides.get("n") is None
        and default_n > 1
        and overrides.get("temperature") is not None
        and float(overrides["temperature"]) == 0.0
    )
    for key, value in overrides.items():
        if value is not None and not (
            key == "temperature" and use_task_sampling_temperature
        ):
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
    if cfg.get("enable_thinking") is not None and not isinstance(
        cfg["enable_thinking"], bool
    ):
        raise ValueError(
            "enable_thinking must be true or false when provided, "
            f"got {cfg['enable_thinking']!r}"
        )
    if cfg["n"] < 1:
        raise ValueError(f"n must be >= 1, got {cfg['n']}")
    if cfg["n"] > 1 and cfg["temperature"] == 0.0:
        raise ValueError("n>1 requires temperature>0. Set --temperature > 0.")
    return cfg


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _tokenizer_getter(
    *,
    backend: GenerationBackend | None,
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


def _record_is_unscored(record: GenerationRecord) -> bool:
    return record.meta.get(_UNSCORED_META_KEY) is True


@lru_cache(maxsize=1)
def _scoring_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _SCORING_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _judge_fingerprint(metric_options: dict[str, Any]) -> str:
    settings = {
        key: value
        for key, value in metric_options.items()
        if key.startswith("judge_") and key not in _JUDGE_TRANSPORT_OPTIONS
    }
    settings.setdefault("judge_backend", "api")
    payload = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _judgment_is_current(record: GenerationRecord, fingerprint: str | None) -> bool:
    return (
        not _record_is_unscored(record)
        and record.meta.get(_JUDGE_META_KEY) == fingerprint
    )


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
            count = count_text_tokens(generation, tokenizer_getter())
        normalized.append(count)
    return normalized


def _ensure_output_token_metadata(
    *,
    output: GenerationOutput,
    tokenizer_getter: Callable[[], Any],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> None:
    if not isinstance(output.meta, dict):
        raise ValueError(f"GenerationOutput meta must be a dict for {output.sample_id}")

    prompt_count = output.meta.get("prompt_token_count")
    if prompt_count is None:
        rendered_prompt = render_prompt_with_chat_template(
            output.prompt,
            tokenizer_getter(),
            chat_template_kwargs,
        )
        prompt_count = count_text_tokens(rendered_prompt, tokenizer_getter())
    elif not _is_token_count(prompt_count):
        raise ValueError(
            f"Invalid prompt_token_count for sample {output.sample_id}: {prompt_count!r}"
        )

    output.meta["prompt_token_count"] = int(prompt_count)
    output.meta["response_token_counts"] = _normalize_response_token_counts(
        output=output,
        tokenizer_getter=tokenizer_getter,
    )
    reasons = output.meta.setdefault("finish_reasons", [None] * len(output.generations))
    if (
        not isinstance(reasons, list)
        or len(reasons) != len(output.generations)
        or any(reason is not None and not isinstance(reason, str) for reason in reasons)
    ):
        raise ValueError(f"Invalid finish_reasons for sample {output.sample_id}")


def _generation_token_meta(output: GenerationOutput, local_idx: int) -> dict[str, Any]:
    prompt_count = output.meta["prompt_token_count"]
    response_counts = output.meta["response_token_counts"]
    if not _is_token_count(prompt_count):
        raise ValueError(f"Invalid prompt_token_count for sample {output.sample_id}")
    if (
        not isinstance(response_counts, list)
        or local_idx >= len(response_counts)
        or not _is_token_count(response_counts[local_idx])
    ):
        raise ValueError(f"Invalid response_token_counts for sample {output.sample_id}")
    return {
        "prompt_token_count": int(prompt_count),
        "response_token_count": int(response_counts[local_idx]),
        "finish_reason": output.meta["finish_reasons"][local_idx],
    }


def _token_usage_summary(records: list[GenerationRecord]) -> dict[str, Any]:
    prompt_counts: list[int] = []
    response_counts: list[int] = []
    completed_counts: list[int] = []
    for record in records:
        prompt_count = record.meta.get("prompt_token_count")
        response_count = record.meta.get("response_token_count")
        if not _is_token_count(prompt_count) or not _is_token_count(response_count):
            raise ValueError(
                f"Missing token counts in record meta for sample {record.sample_id}"
            )
        prompt_counts.append(int(prompt_count))
        response_counts.append(int(response_count))
        # Both backends report EOS / configured stop as "stop". Do not infer
        # completion for legacy records from their length or their score.
        if record.meta.get("finish_reason") == "stop" and not record.error:
            completed_counts.append(int(response_count))

    return {
        "avg_prompt_tokens": sum(prompt_counts) / len(records) if records else None,
        "avg_response_tokens": sum(response_counts) / len(records) if records else None,
        "total_prompt_tokens": sum(prompt_counts),
        "total_response_tokens": sum(response_counts),
        "avg_completed_response_tokens": (
            sum(completed_counts) / len(completed_counts) if completed_counts else None
        ),
        "num_completed_responses": len(completed_counts),
        "total_completed_response_tokens": sum(completed_counts),
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


def _init_score_worker(metrics_path: str) -> None:
    global _SCORE_WORKER_METRICS
    # EvalPlus spawns its checker from here under a 4 GiB RLIMIT_AS; a per-core BLAS
    # pool in that fresh numpy would take most of it. Set before metrics load numpy.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    # Benchmark modules are loaded by path and cannot be pickled into spawned workers.
    _SCORE_WORKER_METRICS = _load_module_from_path(
        "aethereval_worker_metrics", Path(metrics_path)
    )


def _score_in_worker(sample: Sample, generation: str):
    return _normalize_score_generation_result(
        _SCORE_WORKER_METRICS.score_generation(sample, generation)
    )


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
    # Records keep the raw generation; graders only see the post-reasoning answer.
    outputs = [
        dataclasses.replace(
            output, generations=[strip_reasoning(text) for text in output.generations]
        )
        for output in outputs
    ]
    batch_score_fn = getattr(metrics_module, "score_generations_batch", None)
    if callable(batch_score_fn):
        samples = [samples_by_id[output.sample_id] for output in outputs]
        score_options = dict(metric_options)
        if runtime_metric_options:
            score_options.update(runtime_metric_options)
        raw_results = batch_score_fn(samples, outputs, score_options)
        return _normalize_batch_score_results(raw_results=raw_results, outputs=outputs)

    num_proc = int(metric_options.get("num_proc", 1))
    if num_proc < 1:
        raise ValueError("num_proc must be >= 1")
    score_bar = (
        Progress(total=total_records, desc=progress_desc) if total_records > 0 else None
    )
    scored: dict[str, list[tuple[float, bool, Any, dict[str, Any]]]] = {}
    try:
        # SCORE_IN_SUBPROCESS: a checker forked from this process would inherit its memory.
        in_pool = num_proc > 1 or getattr(metrics_module, "SCORE_IN_SUBPROCESS", False)
        if in_pool and total_records:
            workers = min(num_proc, total_records)
            _info(f"{progress_desc}: {workers} CPU scoring processes")
            # Spawn avoids inheriting GPU state; executor workers are non-daemonic,
            # so EvalPlus/LiveCodeBench can still launch their test subprocesses.
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_score_worker,
                initargs=(str(Path(metrics_module.__file__).resolve()),),
            )
            try:
                futures = {
                    executor.submit(
                        _score_in_worker, samples_by_id[output.sample_id], text
                    ): (output.sample_id, index)
                    for output in outputs
                    for index, text in enumerate(output.generations)
                }
                by_index = {}
                for future in as_completed(futures):
                    by_index[futures[future]] = future.result()
                    if score_bar is not None:
                        score_bar.update(1)
            finally:
                # On the first error or Ctrl-C, drop the queued records and wait only
                # for the few already handed to a worker.
                executor.shutdown(wait=True, cancel_futures=True)
            return {
                output.sample_id: [
                    by_index[output.sample_id, index]
                    for index in range(len(output.generations))
                ]
                for output in outputs
            }
        for output in outputs:
            sample = samples_by_id[output.sample_id]
            scored[output.sample_id] = []
            for generation_text in output.generations:
                scored[output.sample_id].append(
                    _normalize_score_generation_result(
                        metrics_module.score_generation(sample, generation_text)
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
        meta["finish_reasons"] = [
            record.meta.get("finish_reason") for record in sample_records
        ]
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


def _outputs_to_records(
    *,
    samples_by_id: dict[str, Sample],
    outputs: list[GenerationOutput],
    gen_indices: dict[str, list[int]],
    prompts: dict[str, list[PromptType]] | None,
    scores: dict[str, list[tuple[float, bool, Any, dict[str, Any]]]] | None,
    judge_fingerprint: str | None,
) -> list[GenerationRecord]:
    # Rescored records pass their saved prompts; unscored records are placeholders.
    records: list[GenerationRecord] = []
    for output in outputs:
        sample = samples_by_id[output.sample_id]
        for local_idx, gen_idx in enumerate(gen_indices[output.sample_id]):
            if scores is None:
                score_value, is_pass, parsed = 0.0, False, None
                record_meta = {_UNSCORED_META_KEY: True}
            else:
                score_value, is_pass, parsed, meta = scores[output.sample_id][local_idx]
                record_meta = dict(meta)
                if judge_fingerprint is not None:
                    record_meta[_JUDGE_META_KEY] = judge_fingerprint
            record_meta.update(_generation_token_meta(output, local_idx))
            records.append(
                GenerationRecord(
                    sample_id=sample.id,
                    gen_idx=gen_idx,
                    prompt=(
                        prompts[output.sample_id][local_idx]
                        if prompts is not None
                        else output.prompt
                    ),
                    generation=output.generations[local_idx],
                    score=score_value,
                    is_pass=is_pass,
                    parsed=parsed,
                    gold=sample.gold,
                    error=None,
                    meta=record_meta,
                )
            )
    return records


def _task_fingerprint(task_module: Any, samples: list[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        payload = [
            sample.id, sample.gold, sample.meta, sample.data,
            _to_chat_prompt(task_module.build_prompt(sample)),
        ]
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _run_single_task(
    *,
    task_name: str,
    task_module: Any,
    metrics_module: Any,
    task_dir: Path,
    backend: GenerationBackend | None,
    task_output_dir: Path,
    gen_overrides: dict[str, Any],
    metric_options: dict[str, Any],
    overwrite: bool,
    run_config_common: dict[str, Any],
    tokenizer_getter: Callable[[], Any],
    generate_only: bool,
    eval_only: bool,
    rescore_existing: bool = True,
) -> dict[str, Any]:
    if generate_only == eval_only:
        raise ValueError("exactly one of generate_only and eval_only must be set")
    phase = phase_name(generate_only=generate_only, eval_only=eval_only)
    metric_options = resolve_task_default_metrics(task_name, metric_options)
    _info(f"[{task_name}] loading task from {task_dir}")
    samples_raw = task_module.load_samples(task_dir)
    samples = [_to_sample(item) for item in samples_raw]
    samples_by_id = {sample.id: sample for sample in samples}
    sample_id_set = set()
    for sample in samples:
        if sample.id in sample_id_set:
            raise ValueError(f"Duplicate sample id in task '{task_name}': {sample.id}")
        sample_id_set.add(sample.id)

    ensure_dir(task_output_dir)
    predictions_path = task_output_dir / "predictions.jsonl"
    summary_path = task_output_dir / "summary.json"
    run_config_path = task_output_dir / "run_config.json"

    prior_run_config: dict[str, Any] = {}
    if not overwrite and run_config_path.exists():
        with run_config_path.open("r", encoding="utf-8") as f:
            loaded_run_config = json.load(f)
        if not isinstance(loaded_run_config, dict):
            raise ValueError(
                f"[{task_name}] existing run_config must be a JSON object: "
                f"{run_config_path}"
            )
        prior_run_config = loaded_run_config

    load_protocol = getattr(task_module, "load_protocol", None)
    protocol = load_protocol(task_dir) if load_protocol is not None else None
    if protocol is not None:
        if "protocol" in prior_run_config and prior_run_config["protocol"] != protocol:
            raise ValueError(
                "Saved evaluation protocol differs; use a new run directory"
            )
        metric_options["_protocol"] = protocol

    saved_gen_cfg = prior_run_config.get("generation_config")
    if eval_only and isinstance(saved_gen_cfg, dict):
        gen_cfg = _merge_generation_config(saved_gen_cfg, {})
        requested_gen_cfg = _merge_generation_config(saved_gen_cfg, gen_overrides)
        conflicts = {
            key: {"saved": gen_cfg.get(key), "requested": requested_gen_cfg.get(key)}
            for key, value in gen_overrides.items()
            if value is not None and requested_gen_cfg.get(key) != gen_cfg.get(key)
        }
        if conflicts:
            raise ValueError(
                f"[{task_name}] eval-only generation overrides conflict with the "
                f"saved run config: {conflicts}"
            )
    else:
        gen_cfg = _merge_generation_config(
            resolve_task_default_gen(task_name), gen_overrides
        )
    chat_template_kwargs = chat_template_kwargs_from_generation_config(gen_cfg)
    task_fingerprint = _task_fingerprint(task_module, samples)
    if predictions_path.exists() and not overwrite:
        if not isinstance(saved_gen_cfg, dict):
            raise ValueError(f"[{task_name}] saved predictions have no generation config; use a new run directory")
        if not eval_only:
            conflicts = {
                key: {"saved": prior_run_config.get(key), "requested": run_config_common.get(key)}
                for key in ("model", "model_name")
                if prior_run_config.get(key) != run_config_common.get(key)
            }
            if saved_gen_cfg != gen_cfg:
                conflicts["generation_config"] = {"saved": saved_gen_cfg, "requested": gen_cfg}
            if conflicts:
                raise ValueError(f"[{task_name}] resume generation settings differ: {conflicts}; use a new run directory")
        saved_fingerprint = prior_run_config.get("task_fingerprint")
        if saved_fingerprint is not None and saved_fingerprint != task_fingerprint:
            raise ValueError(f"[{task_name}] saved task data or prompts differ; use a new run directory")

    n = int(gen_cfg["n"])
    _info(
        f"[{task_name}] samples={len(samples)} n={n} "
        f"temperature={gen_cfg['temperature']} phase={phase} "
        f"overwrite={overwrite} "
        f"data_file={getattr(task_module, 'DATA_FILE', '(unknown)')}"
    )

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
            if prior_run_config.get("task_fingerprint") is None:
                sample = samples_by_id[record.sample_id]
                expected_prompt = _to_chat_prompt(task_module.build_prompt(sample))
                if record.gold != sample.gold or record.prompt != expected_prompt:
                    raise ValueError(f"[{task_name}] saved prediction prompt or gold differs; use a new run directory")
            dedup[(record.sample_id, record.gen_idx)] = record
        existing_records = list(dedup.values())

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

    if generate_only and existing_records and pending_record_count:
        conflicts = {
            key: {"saved": prior_run_config.get(key), "requested": run_config_common.get(key)}
            for key in ("backend", "backend_kwargs")
            if prior_run_config.get(key) != run_config_common.get(key)
        }
        if conflicts:
            raise ValueError(f"[{task_name}] resume generation settings differ: {conflicts}; use a new run directory")

    # Preserve generation provenance when a backend is supplied only for scoring.
    task_run_config = (
        dict(prior_run_config)
        if prior_run_config and (eval_only or not pending_record_count)
        else dict(run_config_common)
    )
    if generate_only:
        # Write before inference so interrupted runs retain their data and settings.
        task_run_config.update({
            "task": task_name,
            "generation_config": gen_cfg,
            "task_fingerprint": task_fingerprint,
        })
        if protocol is not None:
            task_run_config["protocol"] = protocol
        write_json(run_config_path, task_run_config)

    if eval_only and pending_record_count:
        missing_examples = [
            f"{sample.id}:{gen_idx}"
            for sample in samples
            for gen_idx in pending_indices[sample.id]
        ][:10]
        raise ValueError(
            f"[{task_name}] eval-only requires complete existing predictions; "
            f"missing_records={pending_record_count} "
            f"missing_samples={len(pending_inputs)} "
            f"examples={missing_examples}. Run --generate-only first with the "
            "same --model/--model-name, --output-dir, --run-id, tasks, and n."
        )

    if eval_only:
        validate_metrics = getattr(metrics_module, "validate_metric_options", None)
        if callable(validate_metrics):
            validate_metrics({**metric_options, "n": n})

    preserve_existing_scores = bool(
        getattr(metrics_module, "PRESERVE_EXISTING_SCORES_ON_RESUME", False)
    )
    judge_fingerprint = (
        _judge_fingerprint(metric_options) if preserve_existing_scores else None
    )
    rescored_record_count = 0
    if existing_records and eval_only:
        if rescore_existing or not preserve_existing_scores:
            records_to_rescore = list(existing_records)
            preserved_records: list[GenerationRecord] = []
        else:
            # Reuse judgments made with the same judge settings; explicit
            # --eval-only re-judges everything.
            records_to_rescore = [
                record
                for record in existing_records
                if not _judgment_is_current(record, judge_fingerprint)
            ]
            preserved_records = [
                record
                for record in existing_records
                if _judgment_is_current(record, judge_fingerprint)
            ]

        if records_to_rescore:
            _info(
                f"[{task_name}] rescoring existing records ({len(records_to_rescore)})"
            )
            existing_outputs, existing_gen_indices = _records_to_generation_outputs(
                records_to_rescore
            )
            for output in existing_outputs:
                _ensure_output_token_metadata(
                    output=output,
                    tokenizer_getter=tokenizer_getter,
                    chat_template_kwargs=chat_template_kwargs,
                )
            judge_client = metric_options.get("_judge_client")
            if isinstance(judge_client, _LazyJudgeClient):
                # Start here, not in a judge worker thread: the judge's guarded
                # subprocesses exit with the thread that started them.
                judge_client.start()
            runtime_metric_options = {}
            if getattr(metrics_module, "REQUIRES_BACKEND", False):
                runtime_metric_options["_backend"] = backend
            scores = _score_generation_outputs(
                metrics_module=metrics_module,
                samples_by_id=samples_by_id,
                outputs=existing_outputs,
                metric_options={**metric_options, "n": n},
                runtime_metric_options=runtime_metric_options,
                total_records=len(records_to_rescore),
                progress_desc=f"[{task_name}] rescoring",
            )
            rescored_existing = _outputs_to_records(
                samples_by_id=samples_by_id,
                outputs=existing_outputs,
                gen_indices=existing_gen_indices,
                prompts={
                    sample_id: [record.prompt for record in records]
                    for sample_id, records in _group_records_by_sample(
                        records_to_rescore
                    ).items()
                },
                scores=scores,
                judge_fingerprint=judge_fingerprint,
            )

            sample_order = {sample.id: idx for idx, sample in enumerate(samples)}
            existing_records = sorted(
                preserved_records + rescored_existing,
                key=lambda record: (sample_order[record.sample_id], record.gen_idx),
            )
            rescored_record_count = len(rescored_existing)
            write_jsonl(
                predictions_path,
                (_record_to_json(record) for record in existing_records),
            )
            _info(f"[{task_name}] existing-record scoring finished")
        else:
            _info(
                f"[{task_name}] resume: preserving existing scores "
                f"({len(existing_records)})"
            )
    elif not existing_records and predictions_path.exists() and generate_only:
        predictions_path.unlink()

    new_records: list[GenerationRecord] = []

    if pending_inputs:
        if backend is None:
            raise RuntimeError(
                f"[{task_name}] generation requested without an inference backend"
            )
        backend_label = getattr(backend, "name", "backend")
        _info(f"[{task_name}] starting {backend_label} generation")
        custom_generate = getattr(task_module, "generate_outputs", None)
        if callable(custom_generate):
            generated_outputs = custom_generate(
                backend=backend,
                samples=samples,
                pending_indices=pending_indices,
                existing_records=existing_records,
                gen_cfg=gen_cfg,
            )
        else:
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

        for output in generated_outputs:
            _ensure_output_token_metadata(
                output=output,
                tokenizer_getter=tokenizer_getter,
                chat_template_kwargs=chat_template_kwargs,
            )
        new_records = _outputs_to_records(
            samples_by_id=samples_by_id,
            outputs=generated_outputs,
            gen_indices=pending_indices,
            prompts=None,
            scores=None,
            judge_fingerprint=None,
        )
        append_jsonl(predictions_path, (_record_to_json(r) for r in new_records))
        _info(
            f"[{task_name}] generation finished: new_records={len(new_records)} "
            "scored=False"
        )
    else:
        _info(f"[{task_name}] no pending generations; skip inference")

    all_records = existing_records + new_records
    grouped_records = _group_records_by_sample(all_records)
    sample_results = _build_sample_results(samples, grouped_records)
    generation_complete = len(all_records) == len(samples) * n
    unscored_record_count = sum(_record_is_unscored(record) for record in all_records)

    if generate_only:
        metrics: dict[str, Any] = {}
        warnings: list[str] = []
        primary_metric, primary_score = None, None
    else:
        if not generation_complete:
            raise RuntimeError(
                f"[{task_name}] internal error: evaluation reached aggregation with "
                "incomplete generations"
            )
        aggregate_result = metrics_module.aggregate(
            sample_results, {**metric_options, "n": n}
        )
        if not isinstance(aggregate_result, dict):
            raise ValueError("aggregate must return a dict[str, float]")
        raw_warnings = aggregate_result.pop("__warnings__", [])
        warnings = (
            [str(item) for item in raw_warnings]
            if isinstance(raw_warnings, list)
            else [str(raw_warnings)]
        )
        # Scoring replaces every generation placeholder, so a remaining flag was
        # set by the metric (e.g. an unparseable judgment) and aggregate excludes it.
        if unscored_record_count:
            warnings.append(
                f"{unscored_record_count} records were left unscored by the metric "
                "and will be scored again on resume"
            )
        metrics = aggregate_result
        primary_metric, primary_score = _resolve_primary_metric(metrics_module, metrics)

    token_usage = _token_usage_summary(all_records)
    if not generate_only and token_usage["avg_prompt_tokens"] is not None:
        metrics["avg_prompt_tokens"] = token_usage["avg_prompt_tokens"]
    if not generate_only and token_usage["avg_response_tokens"] is not None:
        metrics["avg_response_tokens"] = token_usage["avg_response_tokens"]
    if not generate_only and token_usage["avg_completed_response_tokens"] is not None:
        metrics["avg_completed_response_tokens"] = token_usage[
            "avg_completed_response_tokens"
        ]

    summary = {
        "task": task_name,
        "phase": phase,
        "num_samples": len(samples),
        "n": n,
        "existing_records": len(existing_records),
        "new_records": len(new_records),
        "rescored_records": rescored_record_count,
        "total_records": len(all_records),
        "unscored_records": unscored_record_count,
        "generation_complete": generation_complete,
        "evaluation_complete": generation_complete
        and unscored_record_count == 0
        and not generate_only,
        "metrics": metrics,
        "token_usage": token_usage,
        **primary_score_fields(
            primary_metric, primary_score,
            scale=getattr(metrics_module, "PRIMARY_SCORE_SCALE", primary_score_scale(primary_metric)),
        ),
        "warnings": warnings,
    }
    _info(
        f"[{task_name}] phase done: total_records={len(all_records)} "
        f"unscored_records={unscored_record_count} "
        f"metrics=[{_metric_keys_preview(metrics)}]"
    )
    if warnings:
        _info(f"[{task_name}] warnings={warnings}")

    task_run_config.update(
        {
            "task": task_name,
            "task_dir": str(task_dir),
            "generation_config": gen_cfg,
            "task_fingerprint": task_fingerprint,
            "metric_options": {
                **{
                    key: value
                    for key, value in metric_options.items()
                    if not str(key).startswith("_")
                },
                "n": n,
            },
            "overwrite": overwrite,
            "phase": phase,
        }
    )
    if eval_only:
        task_run_config["scoring_packages"] = dict(_scoring_package_versions())

    if protocol is not None:
        task_run_config["protocol"] = protocol
    write_json(run_config_path, task_run_config)
    if generate_only and not new_records:
        evaluated_summary = _evaluated_summary(summary_path, phase)
        if evaluated_summary is not None:
            _info(f"[{task_name}] nothing to generate; keeping evaluated summary")
            return evaluated_summary
    write_json(summary_path, summary)
    return summary


def _evaluated_summary(path: Path, phase: str) -> dict[str, Any] | None:
    """Return a readable summary.json written by another phase, if any."""
    try:
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(summary, dict) or summary.get("phase") == phase:
        return None
    return summary


def _repeat_generation_overrides(
    task_name: str,
    gen_overrides: dict[str, Any],
    repeat_index: int,
) -> tuple[dict[str, Any], int]:
    overrides = dict(gen_overrides)
    requested_seed = overrides.get("seed")
    if requested_seed is None:
        requested_seed = resolve_task_default_gen(task_name).get("seed")
    base_seed = int(requested_seed) if requested_seed is not None else 0
    seed = base_seed + repeat_index
    overrides["seed"] = seed
    return overrides, seed


def _average_repeat_metrics(
    repeat_summaries: list[dict[str, Any]],
) -> dict[str, float]:
    if not repeat_summaries:
        return {}
    metrics_per_repeat = [summary.get("metrics", {}) for summary in repeat_summaries]
    common_keys = set(metrics_per_repeat[0]).intersection(
        *(set(metrics) for metrics in metrics_per_repeat[1:])
    )
    return {
        str(key): sum(float(metrics[key]) for metrics in metrics_per_repeat)
        / len(metrics_per_repeat)
        for key in sorted(common_keys)
        if all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            for metrics in metrics_per_repeat
        )
    }


def _aggregate_repeat_token_usage(
    repeat_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    total_prompt_tokens = sum(
        int(summary.get("token_usage", {}).get("total_prompt_tokens", 0))
        for summary in repeat_summaries
    )
    total_response_tokens = sum(
        int(summary.get("token_usage", {}).get("total_response_tokens", 0))
        for summary in repeat_summaries
    )
    total_records = sum(
        int(summary.get("total_records", 0)) for summary in repeat_summaries
    )
    completed_tokens = sum(
        int(summary.get("token_usage", {}).get("total_completed_response_tokens", 0))
        for summary in repeat_summaries
    )
    completed_count = sum(
        int(summary.get("token_usage", {}).get("num_completed_responses", 0))
        for summary in repeat_summaries
    )
    return {
        "avg_prompt_tokens": (
            total_prompt_tokens / total_records if total_records else None
        ),
        "avg_response_tokens": (
            total_response_tokens / total_records if total_records else None
        ),
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_response_tokens,
        "avg_completed_response_tokens": (
            completed_tokens / completed_count if completed_count else None
        ),
        "num_completed_responses": completed_count,
        "total_completed_response_tokens": completed_tokens,
    }


def _run_repeated_task(
    *,
    num_repeats: int,
    task_name: str,
    task_module: Any,
    metrics_module: Any,
    task_dir: Path,
    backend: GenerationBackend | None,
    task_output_dir: Path,
    gen_overrides: dict[str, Any],
    metric_options: dict[str, Any],
    overwrite: bool,
    run_config_common: dict[str, Any],
    tokenizer_getter: Callable[[], Any],
    generate_only: bool,
    eval_only: bool,
    rescore_existing: bool = True,
) -> dict[str, Any]:
    phase = phase_name(generate_only=generate_only, eval_only=eval_only)
    if num_repeats == 1:
        summary = _run_single_task(
            task_name=task_name,
            task_module=task_module,
            metrics_module=metrics_module,
            task_dir=task_dir,
            backend=backend,
            task_output_dir=task_output_dir,
            gen_overrides=gen_overrides,
            metric_options=metric_options,
            overwrite=overwrite,
            run_config_common={**run_config_common, "num_repeats": 1},
            tokenizer_getter=tokenizer_getter,
            generate_only=generate_only,
            eval_only=eval_only,
            rescore_existing=rescore_existing,
        )
        if summary.get("phase") == phase:
            summary["num_repeats"] = 1
            write_json(task_output_dir / "summary.json", summary)
        return summary

    repeat_summaries: list[dict[str, Any]] = []
    repeat_metadata: list[dict[str, Any]] = []
    kept_repeats = 0
    for repeat_index in range(num_repeats):
        repeat_number = repeat_index + 1
        repeat_dir = task_output_dir / f"run_{repeat_number:02d}"
        repeat_overrides, seed = _repeat_generation_overrides(
            task_name,
            gen_overrides,
            repeat_index,
        )
        _info(f"[{task_name}] repeat={repeat_number}/{num_repeats} seed={seed}")
        summary = _run_single_task(
            task_name=task_name,
            task_module=task_module,
            metrics_module=metrics_module,
            task_dir=task_dir,
            backend=backend,
            task_output_dir=repeat_dir,
            gen_overrides=repeat_overrides,
            metric_options=metric_options,
            overwrite=overwrite,
            run_config_common={
                **run_config_common,
                "num_repeats": num_repeats,
                "repeat": repeat_number,
            },
            tokenizer_getter=tokenizer_getter,
            generate_only=generate_only,
            eval_only=eval_only,
            rescore_existing=rescore_existing,
        )
        if summary.get("phase") != phase:
            # A kept evaluated summary contributes only its generation data.
            kept_repeats += 1
            summary = {
                **summary,
                "phase": phase,
                "rescored_records": 0,
                "evaluation_complete": False,
                "metrics": {},
                "primary_metric": None,
                "primary_score": None,
                "warnings": [],
            }
        repeat_summaries.append(summary)
        repeat_metadata.append(
            {
                "repeat": repeat_number,
                "seed": seed,
                "output_dir": str(repeat_dir),
                "metrics": summary.get("metrics", {}),
                "primary_metric": summary.get("primary_metric"),
                "primary_score": summary.get("primary_score"),
            }
        )

    metrics = _average_repeat_metrics(repeat_summaries)
    primary_metrics = {
        summary.get("primary_metric")
        for summary in repeat_summaries
        if summary.get("primary_metric") is not None
    }
    if len(primary_metrics) > 1:
        raise ValueError(
            f"[{task_name}] repeated runs produced inconsistent primary metrics: "
            f"{sorted(primary_metrics)}"
        )
    primary_metric = next(iter(primary_metrics), None)
    primary_score = (
        float(metrics[primary_metric])
        if primary_metric is not None and primary_metric in metrics
        else None
    )
    warning_values = [
        str(warning)
        for summary in repeat_summaries
        for warning in summary.get("warnings", [])
    ]
    warnings = list(dict.fromkeys(warning_values))
    token_usage = _aggregate_repeat_token_usage(repeat_summaries)
    if not generate_only and token_usage["avg_completed_response_tokens"] is not None:
        # Repeats can have different completion rates: pool completed responses.
        metrics["avg_completed_response_tokens"] = token_usage[
            "avg_completed_response_tokens"
        ]
    summary = {
        "task": task_name,
        "phase": phase,
        "num_samples": repeat_summaries[0]["num_samples"],
        "n": repeat_summaries[0]["n"],
        "num_repeats": num_repeats,
        "existing_records": sum(
            int(item.get("existing_records", 0)) for item in repeat_summaries
        ),
        "new_records": sum(
            int(item.get("new_records", 0)) for item in repeat_summaries
        ),
        "rescored_records": sum(
            int(item.get("rescored_records", 0)) for item in repeat_summaries
        ),
        "total_records": sum(
            int(item.get("total_records", 0)) for item in repeat_summaries
        ),
        "unscored_records": sum(
            int(item.get("unscored_records", 0)) for item in repeat_summaries
        ),
        "generation_complete": all(
            bool(item.get("generation_complete")) for item in repeat_summaries
        ),
        "evaluation_complete": all(
            bool(item.get("evaluation_complete")) for item in repeat_summaries
        ),
        "metrics": metrics,
        "token_usage": token_usage,
        **primary_score_fields(
            primary_metric, primary_score,
            scale=getattr(metrics_module, "PRIMARY_SCORE_SCALE", primary_score_scale(primary_metric)),
        ),
        "warnings": warnings,
        "repeats": repeat_metadata,
    }
    ensure_dir(task_output_dir)
    summary_path = task_output_dir / "summary.json"
    evaluated_summary = (
        _evaluated_summary(summary_path, phase)
        if kept_repeats == num_repeats
        else None
    )
    if (
        evaluated_summary is not None
        and evaluated_summary.get("num_repeats") == num_repeats
    ):
        # Every repeat kept its evaluated summary, so keep their average too.
        summary = evaluated_summary
    else:
        write_json(summary_path, summary)
    write_json(
        task_output_dir / "run_config.json",
        {
            **run_config_common,
            "task": task_name,
            "num_repeats": num_repeats,
            "repeat_seeds": [item["seed"] for item in repeat_metadata],
            "phase": phase,
        },
    )
    _info(
        f"[{task_name}] repeats complete: num_repeats={num_repeats} "
        f"primary_metric={primary_metric} primary_score={primary_score}"
    )
    return summary


class _LazyJudgeClient:
    """Offline judge that the runner starts only for a task with records to judge.

    An eval phase that reuses every stored judgment never loads the judge.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._client: OfflineJudgeClient | None = None
        self._start_error: BaseException | None = None
        self._closed = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("offline judge is closed")
            if self._start_error is not None:
                raise RuntimeError(
                    "offline judge failed to start"
                ) from self._start_error
            if self._client is not None:
                return
            _info(
                f"offline judge: model={self._config['model']} "
                f"dp_size={self._config['dp_size']} "
                f"tp_size={self._config['tensor_parallel_size']}"
            )
            try:
                self._client = OfflineJudgeClient(**self._config)
            except BaseException as exc:
                self._start_error = exc
                raise

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        client = self._client
        if client is None:
            raise RuntimeError("offline judge is not running")
        return client.complete(messages, **kwargs)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            client, self._client = self._client, None
            if client is not None:
                client.close()


class _LazyBackend:
    """Generation backend that starts its inference servers on first use.

    A phase whose tasks are all generated already (a resumed or repeated run) never
    loads the candidate model. Private attributes such as _tokenizer do not start it.
    """

    def __init__(self, backend_name: str, **kwargs: Any) -> None:
        self.name = normalize_backend_name(backend_name)
        self._kwargs = dict(backend_name=backend_name, **kwargs)
        self._backend: GenerationBackend | None = None

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("_"):
            backend = self.__dict__.get("_backend")
            if backend is None:
                raise AttributeError(attr)
            return getattr(backend, attr)
        if self._backend is None:
            self._backend = create_backend(**self._kwargs)
        return getattr(self._backend, attr)

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None:
            backend.close()


def run_evaluation(
    *,
    model: str,
    tasks: str,
    output_dir: str | Path,
    model_name: str | None = None,
    dp_size: int = 1,
    tensor_parallel_size: int = 1,
    gen_overrides: dict[str, Any] | None = None,
    num_repeats: int | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
    bootstrap_confidence: float = 0.95,
    metric_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    run_id: str | None = None,
    backend_name: str = "vllm",
    backend_kwargs: dict[str, Any] | None = None,
    backend: GenerationBackend | None = None,
    benchmarks_dir: Path | None = None,
    generate_only: bool = False,
    eval_only: bool = False,
) -> dict[str, Any]:
    """Run the generate-only or eval-only phase, or both in that order.

    Without a phase flag, every task is generated before any is evaluated. A
    backend created here is closed before evaluation starts, so candidate, judge
    and reward models are never loaded together; a caller-supplied backend stays
    open for both phases and cannot be combined with a local judge. That
    automatic eval phase reuses judgments made with unchanged judge settings; an
    explicit eval_only re-judges every record.
    """
    if generate_only and eval_only:
        raise ValueError("generate_only and eval_only are mutually exclusive")
    if eval_only and overwrite:
        raise ValueError("eval_only cannot be combined with overwrite")
    judge_backend = str((metric_options or {}).get("judge_backend", "api")).lower()
    if (
        judge_backend == "local"
        and backend is not None
        and not generate_only
        and not eval_only
    ):
        raise ValueError(
            "offline local judging requires disjoint candidate/judge lifecycles. "
            "Close the supplied backend between a generate_only and an eval_only "
            "run_evaluation call, or let run_evaluation create the backend."
        )
    options = dict(
        model=model,
        tasks=tasks,
        output_dir=output_dir,
        model_name=model_name,
        dp_size=dp_size,
        tensor_parallel_size=tensor_parallel_size,
        gen_overrides=gen_overrides,
        num_repeats=num_repeats,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=bootstrap_confidence,
        metric_options=metric_options,
        run_id=run_id,
        backend_name=backend_name,
        backend_kwargs=backend_kwargs,
        benchmarks_dir=benchmarks_dir,
    )
    if generate_only or eval_only:
        return _run_phase(
            **options,
            overwrite=overwrite,
            backend=backend,
            generate_only=generate_only,
            eval_only=eval_only,
            rescore_existing=True,
        )
    _info(
        "two-phase execution: generating all native tasks first, then "
        "restarting in eval-only mode"
    )
    _run_phase(
        **options,
        overwrite=overwrite,
        backend=backend,
        generate_only=True,
        eval_only=False,
        rescore_existing=True,
    )
    return _run_phase(
        **options,
        overwrite=False,
        backend=backend,
        generate_only=False,
        eval_only=True,
        rescore_existing=False,
    )


def _run_phase(
    *,
    model: str,
    tasks: str,
    output_dir: str | Path,
    model_name: str | None,
    dp_size: int,
    tensor_parallel_size: int,
    gen_overrides: dict[str, Any] | None,
    num_repeats: int | None,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_confidence: float,
    metric_options: dict[str, Any] | None,
    overwrite: bool,
    run_id: str | None,
    backend_name: str,
    backend_kwargs: dict[str, Any] | None,
    backend: GenerationBackend | None,
    benchmarks_dir: Path | None,
    generate_only: bool,
    eval_only: bool,
    rescore_existing: bool,
) -> dict[str, Any]:
    phase = phase_name(generate_only=generate_only, eval_only=eval_only)
    effective_model_kwargs = backend_kwargs
    task_root = benchmarks_dir or BENCHMARKS_DIR
    tasks_map = discover_tasks(task_root)
    available = sorted(tasks_map.keys())
    if not available:
        raise RuntimeError(f"No tasks found in {task_root}")

    selected = parse_task_names(tasks, available)
    out_dir = Path(output_dir)
    effective_model_name = model_output_name(model, model_name)
    this_run_id = run_id or effective_model_name
    run_root = run_output_dir(out_dir, model, run_id, model_name)
    ensure_dir(run_root)
    prior_run_summary: dict[str, Any] = {}
    prior_run_summary_path = run_root / "run_summary.json"
    if eval_only and prior_run_summary_path.exists():
        with prior_run_summary_path.open("r", encoding="utf-8") as f:
            loaded_run_summary = json.load(f)
        if not isinstance(loaded_run_summary, dict):
            raise ValueError(
                f"Existing run summary must be a JSON object: {prior_run_summary_path}"
            )
        prior_run_summary = loaded_run_summary
    _info(f"benchmark_root={task_root}")
    _info(f"discovered_tasks={len(available)} selected={selected}")
    _info(
        f"model={model} model_name={effective_model_name} "
        f"backend={backend_name} dp_size={int(dp_size)} "
        f"tp_size={int(tensor_parallel_size)} phase={phase} "
        f"output_dir={out_dir} run_id={this_run_id}"
    )
    if effective_model_kwargs:
        _info(f"backend_model_kwargs={effective_model_kwargs}")
    _info(f"run_output_dir={run_root}")

    created_backend = False
    if backend is None and not eval_only:
        backend = _LazyBackend(
            backend_name,
            model=model,
            dp_size=dp_size,
            tensor_parallel_size=tensor_parallel_size,
            model_kwargs=effective_model_kwargs,
        )
        created_backend = True
    backend_label = getattr(backend, "name", backend_name)
    if eval_only and backend is None:
        saved_backend = prior_run_summary.get("backend")
        if isinstance(saved_backend, str) and saved_backend.strip():
            backend_label = saved_backend
    get_tokenizer = _tokenizer_getter(
        backend=backend,
        model=model,
        model_kwargs=effective_model_kwargs,
    )

    local_judge_client: _LazyJudgeClient | None = None
    local_judge_key: str | None = None
    try:
        run_config_common = {
            "model": model,
            "model_name": effective_model_name,
            "backend": backend_label,
            "dp_size": int(dp_size),
            "tp_size": int(tensor_parallel_size),
            "backend_kwargs": effective_model_kwargs or {},
            "phase": phase,
        }
        resolved_metric_options = {
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_confidence": float(bootstrap_confidence),
        }
        if metric_options:
            resolved_metric_options.update(metric_options)
        task_plan = []
        for task_name in selected:
            bundle = load_task(task_name, task_root)
            task_metric_options = resolve_task_default_metrics(
                task_name, resolved_metric_options
            )
            uses_local_judge = (
                eval_only
                and getattr(bundle.metrics_module, "USES_LLM_JUDGE", False)
                and str(task_metric_options.get("judge_backend", "api")).lower()
                == "local"
            )
            judge_config = None
            if uses_local_judge:
                judge_model = str(task_metric_options.get("judge_model", "")).strip()
                if not judge_model:
                    raise ValueError(
                        f"[{task_name}] offline local judging requires judge_model "
                        "in the task defaults, config, or --judge-model"
                    )
                judge_dp_size = int(task_metric_options.get("judge_dp_size", 1))
                judge_tp_size = int(
                    task_metric_options.get(
                        "judge_tp_size", int(dp_size) * int(tensor_parallel_size)
                    )
                )
                judge_model_kwargs = dict(
                    task_metric_options.get("judge_sglang_args", {})
                )
                judge_config = {
                    "model": judge_model,
                    "dp_size": judge_dp_size,
                    "tensor_parallel_size": judge_tp_size,
                    "model_kwargs": judge_model_kwargs,
                }
            judge_key = (
                json.dumps(judge_config, sort_keys=True, default=str)
                if judge_config is not None
                else None
            )
            task_plan.append(
                (task_name, bundle, task_metric_options, judge_config, judge_key)
            )

        if eval_only and any(item[4] is not None for item in task_plan):
            # Finish non-judge metrics (including GPU RMs) before loading judges.
            # Stable groups reuse one service per configuration without co-residency.
            groups = {None: []}
            for item in task_plan:
                groups.setdefault(item[4], []).append(item)
            task_plan = [item for group in groups.values() for item in group]
            _info(f"local judge task order: {[item[0] for item in task_plan]}")

        summaries: dict[str, Any] = {}
        for (
            task_name, bundle, task_metric_options, judge_config, requested_judge_key
        ) in task_plan:
            _info(f"===== start task: {task_name} =====")
            task_spec = tasks_map[task_name]
            task_output_dir = run_root / task_name
            task_backend = backend
            created_evaluation_backend = False
            if (
                local_judge_client is not None
                and local_judge_key != requested_judge_key
            ):
                local_judge_client.close()
                local_judge_client = None
                local_judge_key = None
            if judge_config is not None:
                if local_judge_client is None:
                    local_judge_client = _LazyJudgeClient(judge_config)
                    local_judge_key = requested_judge_key
                task_metric_options["_judge_client"] = local_judge_client
            if (
                eval_only
                and task_backend is None
                and getattr(bundle.metrics_module, "REQUIRES_BACKEND", False)
            ):
                create_evaluation_backend = getattr(
                    bundle.metrics_module, "create_evaluation_backend", None
                )
                if not callable(create_evaluation_backend):
                    raise ValueError(
                        f"[{task_name}] eval-only metrics require a backend but do "
                        "not provide create_evaluation_backend()."
                    )
                task_backend = create_evaluation_backend(
                    task_metric_options,
                    dp_size=int(dp_size),
                    tensor_parallel_size=int(tensor_parallel_size),
                )
                created_evaluation_backend = True
                _info(
                    f"[{task_name}] eval-only metric backend="
                    f"{getattr(task_backend, 'name', type(task_backend).__name__)}"
                )
            try:
                task_num_repeats = resolve_phase_num_repeats(
                    task_name,
                    task_output_dir / "run_config.json",
                    runtime_override=num_repeats,
                    eval_only=eval_only,
                )
                summary = _run_repeated_task(
                    num_repeats=task_num_repeats,
                    task_name=task_name,
                    task_module=bundle.task_module,
                    metrics_module=bundle.metrics_module,
                    task_dir=task_spec.task_dir,
                    backend=task_backend,
                    task_output_dir=task_output_dir,
                    gen_overrides=gen_overrides or {},
                    metric_options=task_metric_options,
                    overwrite=overwrite,
                    run_config_common=run_config_common,
                    tokenizer_getter=get_tokenizer,
                    generate_only=generate_only,
                    eval_only=eval_only,
                    rescore_existing=rescore_existing,
                )
            finally:
                if created_evaluation_backend and task_backend is not None:
                    task_backend.close()
            summaries[task_name] = summary
            _info(f"===== finish task: {task_name} =====")

        existing_summaries = load_task_summaries(
            run_root,
            allowed_tasks=available,
            skip_tasks=set(selected),
        )
        if existing_summaries:
            _info(
                "including existing task summaries in run-level aggregation: "
                f"{sorted(existing_summaries.keys())}"
            )

        all_task_summaries = {**existing_summaries, **summaries}
        run_summary = build_run_summary(
            run_root=run_root,
            run_id=this_run_id,
            selected_tasks=selected,
            model=model,
            model_name=effective_model_name,
            backend=backend_label,
            phase=phase,
            task_summaries=all_task_summaries,
        )
        _info(f"run_summary_path={run_root / 'run_summary.json'}")
        return run_summary
    finally:
        if local_judge_client is not None:
            local_judge_client.close()
        if created_backend:
            backend.close()


def inspect_prompts(
    *,
    model: str,
    tasks: str,
    backend_kwargs: dict[str, Any] | None = None,
    benchmarks_dir: Path | None = None,
    inspect_limit: int = 5,
    prompt_renderer: Callable[[PromptType], str] | None = None,
    gen_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_root = benchmarks_dir or BENCHMARKS_DIR
    tasks_map = discover_tasks(task_root)
    available = sorted(tasks_map.keys())
    if not available:
        raise RuntimeError(f"No tasks found in {task_root}")

    selected = parse_task_names(tasks, available)
    limit = max(1, int(inspect_limit))
    _info(f"inspect mode: model={model} tasks={selected} limit={limit}")

    tokenizer = (
        load_chat_tokenizer(model, backend_kwargs) if prompt_renderer is None else None
    )

    task_results: dict[str, list[dict[str, Any]]] = {}
    for task_name in selected:
        bundle = load_task(task_name, task_root)
        task_spec = tasks_map[task_name]
        samples_raw = bundle.task_module.load_samples(task_spec.task_dir)
        samples = [_to_sample(item) for item in samples_raw]
        gen_cfg = _merge_generation_config(
            resolve_task_default_gen(task_name), gen_overrides or {}
        )
        chat_template_kwargs = chat_template_kwargs_from_generation_config(gen_cfg)

        rows: list[dict[str, Any]] = []
        for sample in samples[:limit]:
            prompt = _to_chat_prompt(bundle.task_module.build_prompt(sample))
            if prompt_renderer is None:
                rendered = render_prompt_with_chat_template(
                    prompt,
                    tokenizer,
                    chat_template_kwargs,
                )
            else:
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
