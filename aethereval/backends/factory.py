from typing import Any

from .base import GenerationBackend
from .sglang import SGLangBackend
from .vllm import VLLMBackend


SUPPORTED_BACKENDS = ("vllm", "sglang")


def normalize_backend_name(name: str | None) -> str:
    backend_name = (name or "vllm").strip().lower()
    if backend_name not in SUPPORTED_BACKENDS:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(
            f"Unsupported backend '{name}'. Supported backends: {supported}"
        )
    return backend_name


def create_backend(
    *,
    backend_name: str | None,
    model: str,
    dp_size: int = 1,
    tensor_parallel_size: int = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> GenerationBackend:
    normalized = normalize_backend_name(backend_name)
    if normalized == "vllm":
        return VLLMBackend(
            model=model,
            dp_size=dp_size,
            tensor_parallel_size=tensor_parallel_size,
            model_kwargs=model_kwargs,
        )
    if normalized == "sglang":
        return SGLangBackend(
            model=model,
            dp_size=dp_size,
            tensor_parallel_size=tensor_parallel_size,
            model_kwargs=model_kwargs,
        )
    raise AssertionError(f"Unhandled backend: {normalized}")
