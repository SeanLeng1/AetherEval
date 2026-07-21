"""Batched offline LLM-judge inference using an AetherEval generation backend."""

import json
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from aethereval.backends import (
    GenerationBackend,
    chat_template_kwargs_from_generation_config,
    create_backend,
    validate_system_role_support,
)
from aethereval.core.types import GenerationInput


@dataclass
class _JudgeRequest:
    request_id: int
    messages: list[dict[str, str]]
    gen_cfg: dict[str, Any]
    completed: threading.Event = field(default_factory=threading.Event)
    result: str | None = None
    error: BaseException | None = None


_STOP = object()


class OfflineJudgeClient:
    """Coalesce synchronous judge calls into offline generation batches."""

    name = "offline-sglang-judge"

    def __init__(
        self,
        *,
        model: str,
        dp_size: int,
        tensor_parallel_size: int,
        model_kwargs: dict[str, Any] | None = None,
        batch_size: int = 64,
        batch_wait_seconds: float = 0.01,
        default_max_tokens: int = 4096,
        enable_thinking: bool | None = False,
        backend: GenerationBackend | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = str(model)
        self.batch_size = int(batch_size)
        self.batch_wait_seconds = float(batch_wait_seconds)
        self.default_max_tokens = int(default_max_tokens)
        self.enable_thinking = enable_thinking
        if not self.model.strip():
            raise ValueError("offline judge model cannot be empty")
        if self.batch_size < 1:
            raise ValueError("offline judge batch_size must be >= 1")
        if self.batch_wait_seconds < 0:
            raise ValueError("offline judge batch_wait_seconds must be >= 0")
        if self.default_max_tokens < 1:
            raise ValueError("offline judge default_max_tokens must be >= 1")
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise ValueError("offline judge enable_thinking must be true or false")

        resolved_model_kwargs = dict(model_kwargs or {})
        self._backend = backend or create_backend(
            backend_name="sglang",
            model=self.model,
            dp_size=int(dp_size),
            tensor_parallel_size=int(tensor_parallel_size),
            model_kwargs=resolved_model_kwargs,
        )
        self._chat_tokenizer = (
            tokenizer
            if tokenizer is not None
            else getattr(self._backend, "_tokenizer", None)
        )
        self._validated_system_template_configs: set[str] = set()
        self._system_template_lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._request_id = 0
        self._id_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._batch_worker,
            name="aethereval-offline-judge",
            daemon=True,
        )
        self._thread.start()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        if self._closed:
            raise RuntimeError("offline judge client is closed")
        with self._id_lock:
            request_id = self._request_id
            self._request_id += 1

        gen_cfg: dict[str, Any] = {
            "n": 1,
            "temperature": 1.0 if temperature is None else float(temperature),
            "max_new_tokens": (
                self.default_max_tokens if max_tokens is None else int(max_tokens)
            ),
            "top_p": 1.0 if top_p is None else float(top_p),
            "top_k": -1,
            "_show_progress": False,
        }
        if seed is not None:
            gen_cfg["seed"] = int(seed)
        if self.enable_thinking is not None:
            gen_cfg["enable_thinking"] = self.enable_thinking

        supported_extra = {
            "top_k",
            "min_p",
            "stop",
            "enable_thinking",
            "regex",
            "json_schema",
            "ebnf",
            "structural_tag",
        }
        for key, value in (extra_body or {}).items():
            if key not in supported_extra:
                raise ValueError(
                    f"offline judge does not support extra_body option {key!r}"
                )
            gen_cfg[key] = value

        if any(message.get("role") == "system" for message in messages):
            self._validate_system_role(gen_cfg)

        request = _JudgeRequest(
            request_id=request_id,
            messages=[dict(message) for message in messages],
            gen_cfg=gen_cfg,
        )
        self._queue.put(request)
        request.completed.wait()
        if request.error is not None:
            raise RuntimeError(
                f"offline judge request {request_id} failed: {request.error}"
            ) from request.error
        if request.result is None:
            raise RuntimeError(f"offline judge request {request_id} returned no result")
        return request.result

    def _validate_system_role(self, gen_cfg: dict[str, Any]) -> None:
        template_kwargs = chat_template_kwargs_from_generation_config(gen_cfg)
        cache_key = json.dumps(template_kwargs, sort_keys=True, default=str)
        with self._system_template_lock:
            if cache_key in self._validated_system_template_configs:
                return
            if self._chat_tokenizer is not None:
                validate_system_role_support(
                    self._chat_tokenizer,
                    model=self.model,
                    chat_template_kwargs=template_kwargs,
                )
            else:
                backend_validator = getattr(
                    self._backend, "validate_system_role_support", None
                )
                if not callable(backend_validator):
                    raise ValueError(
                        f"Judge model {self.model!r} system-role compatibility "
                        "cannot be validated because its tokenizer is unavailable."
                    )
                backend_validator(template_kwargs)
            self._validated_system_template_configs.add(cache_key)

    def _batch_worker(self) -> None:
        stop_after_batch = False
        while not stop_after_batch:
            first = self._queue.get()
            if first is _STOP:
                return
            batch = [first]
            deadline = time.monotonic() + self.batch_wait_seconds
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is _STOP:
                    stop_after_batch = True
                    break
                batch.append(item)
            self._process_batch(batch)

    def _process_batch(self, requests: list[_JudgeRequest]) -> None:
        grouped: dict[str, list[_JudgeRequest]] = defaultdict(list)
        configs: dict[str, dict[str, Any]] = {}
        for request in requests:
            key = json.dumps(request.gen_cfg, sort_keys=True, default=str)
            grouped[key].append(request)
            configs[key] = request.gen_cfg

        for key, compatible_requests in grouped.items():
            try:
                inputs = [
                    GenerationInput(
                        sample_id=str(request.request_id),
                        prompt=request.messages,
                        num_generations=1,
                    )
                    for request in compatible_requests
                ]
                outputs = self._backend.generate(inputs, configs[key])
                by_id = {output.sample_id: output for output in outputs}
                if len(by_id) != len(compatible_requests):
                    raise ValueError(
                        "offline judge returned an unexpected number of outputs: "
                        f"{len(by_id)} != {len(compatible_requests)}"
                    )
                for request in compatible_requests:
                    output = by_id[str(request.request_id)]
                    if output.error is not None:
                        raise RuntimeError(output.error)
                    if len(output.generations) != 1:
                        raise ValueError(
                            "offline judge expected exactly one generation for "
                            f"request {request.request_id}"
                        )
                    request.result = str(output.generations[0])
            except BaseException as exc:
                for request in compatible_requests:
                    request.error = exc
            finally:
                for request in compatible_requests:
                    request.completed.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join()
        self._backend.close()


__all__ = ["OfflineJudgeClient"]
