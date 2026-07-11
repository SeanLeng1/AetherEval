from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationInput, GenerationOutput

from ..prompt import (
    _prompt_to_text,
    count_text_tokens,
    count_token_ids,
    load_chat_tokenizer,
)


def _build_sampling_params(gen_cfg: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 256)),
        "temperature": float(gen_cfg.get("temperature", 0.0)),
        "top_p": float(gen_cfg.get("top_p", 1.0)),
    }
    top_k = gen_cfg.get("top_k")
    if top_k is not None and int(top_k) >= 0:
        params["top_k"] = int(top_k)
    if gen_cfg.get("min_p") is not None:
        params["min_p"] = float(gen_cfg["min_p"])
    if gen_cfg.get("stop") is not None:
        params["stop"] = gen_cfg["stop"]
    if gen_cfg.get("seed") is not None:
        params["seed"] = int(gen_cfg["seed"])
    return params


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        for key in ("text", "output_text", "generated_text"):
            if key in output:
                return str(output[key])
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"])
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return _extract_text(outputs[0])

    text = getattr(output, "text", None)
    if text is not None:
        return str(text)

    raise TypeError(f"Unsupported SGLang output type: {type(output).__name__}")


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _dict_first_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _maybe_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _maybe_count_token_ids(token_ids: Any) -> int | None:
    if token_ids is None:
        return None
    try:
        return count_token_ids(token_ids)
    except TypeError:
        return None


def _extract_output_token_count(output: Any) -> int | None:
    if isinstance(output, str):
        return None

    if isinstance(output, dict):
        for key in ("output_token_ids", "output_ids", "token_ids"):
            count = _maybe_count_token_ids(output.get(key))
            if count is not None:
                return count
        count = _dict_first_int(
            output,
            (
                "output_token_count",
                "completion_token_count",
                "num_output_tokens",
                "num_completion_tokens",
                "completion_tokens",
            ),
        )
        if count is not None:
            return count
        meta_info = output.get("meta_info")
        if isinstance(meta_info, dict):
            count = _dict_first_int(
                meta_info,
                (
                    "output_token_count",
                    "completion_token_count",
                    "num_output_tokens",
                    "num_completion_tokens",
                    "completion_tokens",
                ),
            )
            if count is not None:
                return count
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            return _extract_output_token_count(choices[0])
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return _extract_output_token_count(outputs[0])
        return None

    for attr in ("output_token_ids", "output_ids", "token_ids"):
        count = _maybe_count_token_ids(getattr(output, attr, None))
        if count is not None:
            return count
    for attr in (
        "output_token_count",
        "completion_token_count",
        "num_output_tokens",
        "num_completion_tokens",
        "completion_tokens",
    ):
        count = _maybe_int(getattr(output, attr, None))
        if count is not None:
            return count
    meta_info = getattr(output, "meta_info", None)
    if isinstance(meta_info, dict):
        return _dict_first_int(
            meta_info,
            (
                "output_token_count",
                "completion_token_count",
                "num_output_tokens",
                "num_completion_tokens",
                "completion_tokens",
            ),
        )
    return None


def _normalize_outputs(outputs: Any) -> list[Any]:
    if isinstance(outputs, dict):
        return [outputs]
    if isinstance(outputs, list):
        return outputs
    try:
        return list(outputs)
    except TypeError:
        return [outputs]


def _resolve_generation_batch_size(raw_value: Any) -> int:
    batch_size = int(raw_value if raw_value is not None else 128)
    if batch_size < 1:
        raise ValueError(f"generation_batch_size must be >= 1, got {batch_size}")
    return batch_size


def _make_progress_bar(total: int, desc: str, enabled: bool) -> Any:
    if not enabled or total <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit="gen", dynamic_ncols=True)


def _split_payloads(
    payloads: list[dict[str, Any]],
    num_workers: int,
) -> list[list[dict[str, Any]]]:
    worker_payloads: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for idx, payload in enumerate(payloads):
        worker_payloads[idx % num_workers].append(payload)
    return worker_payloads


def _outputs_from_dicts(output_dicts: list[dict[str, Any]]) -> list[GenerationOutput]:
    return [
        GenerationOutput(
            sample_id=item["sample_id"],
            prompt=item["prompt"],
            generations=item["generations"],
            error=item["error"],
            meta=item.get("meta", {}),
        )
        for item in output_dicts
    ]


def _run_generation(
    engine: Any,
    tokenizer: Any,
    payloads: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
    batch_size: int = 128,
    show_progress: bool = True,
    progress_desc: str = "sglang generating",
) -> list[dict[str, Any]]:
    if not payloads:
        return []

    batch_size = _resolve_generation_batch_size(batch_size)
    bucketed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payloads:
        bucketed[int(item["num_generations"])].append(item)

    result_by_index: dict[int, dict[str, Any]] = {}
    sampling_params = _build_sampling_params(gen_cfg)
    progress_total = sum(int(item["num_generations"]) for item in payloads)
    progress_bar = _make_progress_bar(progress_total, progress_desc, show_progress)

    try:
        for n, items in bucketed.items():
            prompts: list[str] = []
            request_payloads: list[dict[str, Any]] = []
            for item in items:
                rendered = _prompt_to_text(item["prompt"], tokenizer)
                for _ in range(n):
                    prompts.append(rendered)
                    request_payloads.append(item)

            grouped_texts: dict[int, list[str]] = defaultdict(list)
            grouped_token_counts: dict[int, list[int | None]] = defaultdict(list)
            prompt_token_counts: dict[int, int] = {}
            for start in range(0, len(prompts), batch_size):
                end = start + batch_size
                batch_prompts = prompts[start:end]
                batch_payloads = request_payloads[start:end]
                outputs = _normalize_outputs(
                    engine.generate(batch_prompts, sampling_params)
                )
                if len(outputs) != len(batch_payloads):
                    raise RuntimeError(
                        f"SGLang returned {len(outputs)} outputs for {len(batch_payloads)} prompts."
                    )
                for item, prompt_text, output in zip(
                    batch_payloads, batch_prompts, outputs, strict=True
                ):
                    item_idx = int(item["idx"])
                    grouped_texts[item_idx].append(_extract_text(output))
                    grouped_token_counts[item_idx].append(
                        _extract_output_token_count(output)
                    )
                    prompt_token_counts[item_idx] = count_text_tokens(
                        prompt_text, tokenizer
                    )
                if progress_bar is not None:
                    progress_bar.update(len(batch_payloads))

            for item in items:
                item_idx = int(item["idx"])
                texts = grouped_texts[item_idx]
                if len(texts) != n:
                    raise RuntimeError(
                        f"SGLang returned {len(texts)} candidates for sample {item['sample_id']}; expected {n}."
                    )
                result_by_index[item["idx"]] = {
                    "idx": item["idx"],
                    "sample_id": item["sample_id"],
                    "prompt": item["prompt"],
                    "generations": texts,
                    "error": None,
                    "meta": {
                        "prompt_token_count": prompt_token_counts[item_idx],
                        "response_token_counts": grouped_token_counts[item_idx],
                    },
                }
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return [result_by_index[i] for i in sorted(result_by_index)]


def _release_engine_memory(engine: Any, memory_saver_enabled: bool) -> None:
    """Free the engine's weights + KV cache so scoring models can use its GPUs."""
    if not memory_saver_enabled:
        raise RuntimeError(
            "engine memory offload requires enable_memory_saver=true "
            "(default; was overridden via model_kwargs)"
        )
    release = getattr(engine, "release_memory_occupation", None)
    if not callable(release):
        raise RuntimeError(
            "sglang Engine has no release_memory_occupation(); this sglang build "
            "cannot offload for RM scoring"
        )
    release()


def _resume_engine_memory(engine: Any, model_path: str) -> None:
    """Re-occupy GPU memory and reload weights (release does not preserve their contents)."""
    resume = getattr(engine, "resume_memory_occupation", None)
    update_weights = getattr(engine, "update_weights_from_disk", None)
    if not callable(resume) or not callable(update_weights):
        raise RuntimeError(
            "sglang Engine lacks resume_memory_occupation()/update_weights_from_disk(); "
            "cannot restore the engine after RM scoring"
        )
    resume()
    update_weights(model_path)


def _score_with_engine_offloaded(
    *,
    engine: Any,
    memory_saver_enabled: bool,
    model_path: str,
    num_devices: int,
    model_paths: list[str],
    conversations: list[list[dict[str, str]]],
    scorer_kwargs: dict[str, Any],
) -> dict[str, list[float]]:
    """Offload one engine, score its shard across its visible devices, resume it."""
    from benchmark_utils.reward_model import score_conversations_sharded

    devices = [f"cuda:{i}" for i in range(num_devices)]
    _release_engine_memory(engine, memory_saver_enabled)
    try:
        return score_conversations_sharded(
            model_paths=model_paths,
            conversations=conversations,
            devices=devices,
            scorer_kwargs=scorer_kwargs,
        )
    finally:
        _resume_engine_memory(engine, model_path)


def _split_contiguous(items: list[Any], num_chunks: int) -> list[list[Any]]:
    base, extra = divmod(len(items), num_chunks)
    chunks: list[list[Any]] = []
    start = 0
    for index in range(num_chunks):
        size = base + (1 if index < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


class SGLangBackend:
    """SGLang offline backend.

    - dp_size = 1: single SGLang Engine.
    - dp_size > 1: Ray data-parallel workers, one independent Engine each
      (sglang's native dp_size router is much slower for offline batches).
    RM scoring offloads each engine (memory saver) and shards conversations
    over all dp*tp GPUs, then resumes + reloads weights from disk.
    """

    name = "sglang"

    def __init__(
        self,
        model: str,
        dp_size: int = 1,
        tensor_parallel_size: int = 1,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.dp_size = int(dp_size)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.model_kwargs = dict(model_kwargs or {})
        self.generation_batch_size = _resolve_generation_batch_size(
            self.model_kwargs.pop("generation_batch_size", None)
        )

        self._sglang = None
        self._ray = None
        self._engine = None
        self._tokenizer = None
        self._workers: list[Any] = []
        self._memory_saver_enabled = False

        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )

        if self.dp_size == 1:
            self._init_single()
        else:
            self._init_ray()

    def _import_sglang(self) -> Any:
        try:
            import sglang
        except ImportError as exc:
            raise RuntimeError(
                "sglang is not installed. Install dependencies first (`pip install -e .[sglang]`)."
            ) from exc
        self._sglang = sglang
        return sglang

    def _engine_args(self) -> dict[str, Any]:
        engine_args: dict[str, Any] = {
            "model_path": self.model,
            "tp_size": self.tensor_parallel_size,
        }
        engine_args.update(
            {
                key: value
                for key, value in self.model_kwargs.items()
                if key != "generation_batch_size"
            }
        )
        # release/resume (RM-scoring offload) needs the memory saver; overridable via model_kwargs.
        engine_args.setdefault("enable_memory_saver", True)
        return engine_args

    def _init_single(self) -> None:
        sglang = self._import_sglang()
        engine_args = self._engine_args()
        self._memory_saver_enabled = bool(engine_args.get("enable_memory_saver", False))
        self._engine = sglang.Engine(**engine_args)
        self._tokenizer = load_chat_tokenizer(self.model, self.model_kwargs)

    def _init_ray(self) -> None:
        self._import_sglang()
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError(
                "ray is not installed. Install dependencies first (`pip install -e .`)."
            ) from exc

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._ray = ray
        engine_args = self._engine_args()
        self._memory_saver_enabled = bool(engine_args.get("enable_memory_saver", False))
        model = self.model
        model_kwargs = self.model_kwargs
        generation_batch_size = self.generation_batch_size
        tensor_parallel_size = self.tensor_parallel_size
        memory_saver_enabled = self._memory_saver_enabled

        @ray.remote(num_gpus=self.tensor_parallel_size)
        class _Worker:
            def __init__(
                self,
                _engine_args: dict[str, Any],
                _model: str,
                _model_kwargs: dict[str, Any],
                _generation_batch_size: int,
            ) -> None:
                import sglang as _sglang

                from aethereval.backends.prompt import (
                    load_chat_tokenizer as _load_chat_tokenizer,
                )

                self._engine = _sglang.Engine(**_engine_args)
                self._model = _model
                self._tokenizer = _load_chat_tokenizer(_model, _model_kwargs)
                self._generation_batch_size = _generation_batch_size

            def generate(
                self,
                payloads: list[dict[str, Any]],
                gen_cfg: dict[str, Any],
            ) -> list[dict[str, Any]]:
                return _run_generation(
                    engine=self._engine,
                    tokenizer=self._tokenizer,
                    payloads=payloads,
                    gen_cfg=gen_cfg,
                    batch_size=self._generation_batch_size,
                )

            def score_reward_models_shard(
                self,
                model_paths: list[str],
                conversations: list[list[dict[str, str]]],
                scorer_kwargs: dict[str, Any],
            ) -> dict[str, list[float]]:
                # Ray scopes CUDA_VISIBLE_DEVICES to this worker's tp GPUs.
                return _score_with_engine_offloaded(
                    engine=self._engine,
                    memory_saver_enabled=memory_saver_enabled,
                    model_path=self._model,
                    num_devices=tensor_parallel_size,
                    model_paths=model_paths,
                    conversations=conversations,
                    scorer_kwargs=scorer_kwargs,
                )

            def close(self) -> None:
                if self._engine is None:
                    return
                shutdown = getattr(self._engine, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                else:
                    close = getattr(self._engine, "close", None)
                    if callable(close):
                        close()
                self._engine = None
                self._tokenizer = None

        self._workers = [
            _Worker.remote(engine_args, model, model_kwargs, generation_batch_size)
            for _ in range(self.dp_size)
        ]

    def generate(
        self,
        inputs: list[GenerationInput],
        gen_cfg: dict[str, Any],
    ) -> list[GenerationOutput]:
        payloads = [
            {
                "idx": idx,
                "sample_id": item.sample_id,
                "prompt": item.prompt,
                "num_generations": int(item.num_generations),
            }
            for idx, item in enumerate(inputs)
        ]
        if not payloads:
            return []

        if self.dp_size == 1:
            assert self._engine is not None
            output_dicts = _run_generation(
                engine=self._engine,
                tokenizer=self._tokenizer,
                payloads=payloads,
                gen_cfg=gen_cfg,
                batch_size=self.generation_batch_size,
            )
        else:
            assert self._ray is not None
            worker_payloads = _split_payloads(payloads, len(self._workers))
            refs = []
            for worker, worker_items in zip(self._workers, worker_payloads):
                refs.append(worker.generate.remote(worker_items, gen_cfg))
            nested = self._ray.get(refs)
            output_dicts = [item for sublist in nested for item in sublist]
            output_dicts.sort(key=lambda x: x["idx"])

        return _outputs_from_dicts(output_dicts)

    def score_reward_models(
        self,
        model_paths: list[str],
        conversations: list[list[dict[str, str]]],
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        """Offload the engine(s), score with each reward model data-parallel, resume.

        Scoring reuses the run's GPU budget (dp_size * tp_size): each engine is
        released, its conversations shard is scored one process per GPU (one
        reward model resident at a time), then it resumes + reloads weights.
        """
        unique_paths = list(dict.fromkeys(model_paths))
        kwargs = dict(scorer_kwargs or {})
        if not conversations:
            return {path: [] for path in unique_paths}

        if self.dp_size == 1:
            assert self._engine is not None
            return _score_with_engine_offloaded(
                engine=self._engine,
                memory_saver_enabled=self._memory_saver_enabled,
                model_path=self.model,
                num_devices=self.tensor_parallel_size,
                model_paths=unique_paths,
                conversations=conversations,
                scorer_kwargs=kwargs,
            )

        assert self._ray is not None
        num_shards = min(len(self._workers), len(conversations))
        shards = _split_contiguous(conversations, num_shards)
        refs = [
            self._workers[shard_index].score_reward_models_shard.remote(
                unique_paths, shard, kwargs
            )
            for shard_index, shard in enumerate(shards)
        ]
        shard_results = self._ray.get(refs)
        merged: dict[str, list[float]] = {path: [] for path in unique_paths}
        for shard_result in shard_results:
            for path in unique_paths:
                merged[path].extend(shard_result[path])
        for path in unique_paths:
            if len(merged[path]) != len(conversations):
                raise ValueError(
                    f"sharded scoring returned {len(merged[path])} scores for {path}, "
                    f"expected {len(conversations)}"
                )
        return merged

    def close(self) -> None:
        if self._engine is not None:
            shutdown = getattr(self._engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                close = getattr(self._engine, "close", None)
                if callable(close):
                    close()
        self._engine = None
        self._tokenizer = None
        if self._ray is not None and self._workers:
            self._ray.get([worker.close.remote() for worker in self._workers])
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
        self._workers = []
