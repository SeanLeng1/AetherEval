"""LiteLLM client shared by native LLM-judge benchmarks."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

import litellm
from tqdm.auto import tqdm

T = TypeVar("T")
R = TypeVar("R")
NORMAL_FORMAT_ATTEMPTS = 3


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    base_url: str | None
    api_key: str | None
    workers: int
    timeout: float
    max_retries: int
    temperature: float | None
    max_new_tokens: int | None
    top_p: float | None
    enable_thinking: bool | None
    local_client: Any | None = None


def resolve_judge_settings(
    metric_options: dict[str, Any] | None,
    *,
    default_model: str,
) -> JudgeSettings:
    options = metric_options or {}
    model = str(options.get("judge_model", default_model)).strip()
    if not model:
        raise ValueError("judge model cannot be empty")

    local_client = options.get("_judge_client")
    if local_client is None:
        raw_base_url = (
            options.get("judge_base_url")
            or os.environ.get("AETHEREVAL_JUDGE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        base_url = str(raw_base_url).rstrip("/") if raw_base_url else None

        explicit_key_env = options.get("judge_api_key_env")
        if explicit_key_env is not None:
            key_env = str(explicit_key_env).strip()
            if not key_env:
                raise ValueError("judge API-key environment variable cannot be empty")
            api_key = os.environ.get(key_env)
            if not api_key:
                raise RuntimeError(
                    f"LLM-judge API key environment variable {key_env} is not set."
                )
        else:
            api_key = os.environ.get("AETHEREVAL_JUDGE_API_KEY")
        if api_key is None and base_url is not None:
            api_key = os.environ.get("OPENAI_API_KEY")
    else:
        base_url = "offline://local-judge"
        api_key = "-"

    workers = int(options.get("judge_workers", 64))
    timeout = float(options.get("judge_timeout", 300.0))
    max_retries = int(options.get("judge_max_retries", 5))
    raw_temperature = options.get("judge_temperature")
    temperature = None if raw_temperature is None else float(raw_temperature)
    raw_max_new_tokens = options.get("judge_max_new_tokens")
    max_new_tokens = None if raw_max_new_tokens is None else int(raw_max_new_tokens)
    raw_top_p = options.get("judge_top_p")
    top_p = None if raw_top_p is None else float(raw_top_p)
    enable_thinking = options.get("judge_enable_thinking")
    if workers < 1:
        raise ValueError("judge_workers must be >= 1")
    if timeout <= 0:
        raise ValueError("judge_timeout must be > 0")
    if max_retries < 0:
        raise ValueError("judge_max_retries must be >= 0")
    if temperature is not None and temperature < 0:
        raise ValueError("judge_temperature must be >= 0")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("judge_max_new_tokens must be >= 1")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("judge_top_p must be in (0, 1]")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise ValueError("judge_enable_thinking must be true or false")

    return JudgeSettings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        workers=workers,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        enable_thinking=enable_thinking,
        local_client=local_client,
    )


def chat_completion(
    settings: JudgeSettings,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    effective_temperature = (
        settings.temperature if temperature is None else float(temperature)
    )
    effective_max_tokens = (
        settings.max_new_tokens if max_tokens is None else int(max_tokens)
    )
    effective_top_p = settings.top_p if top_p is None else float(top_p)
    effective_extra_body = dict(extra_body or {})

    if settings.local_client is not None:
        if settings.enable_thinking is not None:
            effective_extra_body.setdefault("enable_thinking", settings.enable_thinking)
        return settings.local_client.complete(
            messages,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
            top_p=effective_top_p,
            seed=seed,
            extra_body=effective_extra_body or None,
        )

    if settings.enable_thinking is not None:
        raw_template_kwargs = effective_extra_body.get("chat_template_kwargs", {})
        if not isinstance(raw_template_kwargs, dict):
            raise ValueError("extra_body.chat_template_kwargs must be a mapping/object")
        template_kwargs = dict(raw_template_kwargs)
        template_kwargs.setdefault("enable_thinking", settings.enable_thinking)
        effective_extra_body["chat_template_kwargs"] = template_kwargs

    model = settings.model
    if settings.base_url is not None and not model.startswith("openai/"):
        # --judge-base-url is explicitly an OpenAI-compatible endpoint. The
        # prefix selects LiteLLM's OpenAI transport while the provider prefix is
        # stripped from the model name sent to that endpoint.
        model = f"openai/{model}"

    request_options: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": settings.timeout,
        "num_retries": 0,
    }
    if effective_temperature is not None:
        request_options["temperature"] = effective_temperature
    if effective_max_tokens is not None:
        request_options["max_tokens"] = effective_max_tokens
    if effective_top_p is not None:
        request_options["top_p"] = effective_top_p
    if seed is not None:
        request_options["seed"] = int(seed)
    if effective_extra_body:
        request_options["extra_body"] = effective_extra_body
    if settings.base_url is not None:
        request_options["base_url"] = settings.base_url
    if settings.api_key is not None:
        request_options["api_key"] = settings.api_key

    last_error: BaseException | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            return _extract_content(litellm.completion(**request_options))
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "status_code", None)
            if (
                isinstance(status_code, int)
                and status_code < 500
                and status_code not in {408, 409, 429}
            ):
                raise RuntimeError(f"judge request failed: {exc}") from exc

        if attempt < settings.max_retries:
            time.sleep(min(2**attempt, 16))

    raise RuntimeError(
        f"judge request failed after {settings.max_retries + 1} attempts: {last_error}"
    ) from last_error


def local_constraint_body(
    settings: JudgeSettings,
    *,
    regex: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build an SGLang structured-output request for a local judge only."""

    if settings.local_client is None:
        return None
    if (regex is None) == (json_schema is None):
        raise ValueError("exactly one local judge constraint must be provided")
    if regex is not None:
        return {"regex": regex}
    return {
        "json_schema": json.dumps(
            json_schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    }


def parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    desc: str,
) -> list[R]:
    values = list(items)
    if not values:
        return []

    progress = tqdm(
        total=len(values),
        desc=desc,
        unit="judge",
        dynamic_ncols=True,
        mininterval=1.0,
    )

    results: list[R | None] = [None] * len(values)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fn, value): idx for idx, value in enumerate(values)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                progress.update(1)
    finally:
        progress.close()

    if any(result is None for result in results):
        raise RuntimeError("judge worker returned an incomplete result set")
    return [result for result in results if result is not None]


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def _extract_content(response: Any) -> str:
    if not isinstance(response, dict) and hasattr(response, "model_dump"):
        response = response.model_dump()
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected judge response schema: {response}") from exc
    if isinstance(content, str):
        if not content.strip():
            raise ValueError("judge returned empty content")
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        joined = "".join(parts)
        if joined.strip():
            return joined
    raise ValueError(f"judge returned unsupported message content: {content!r}")


__all__ = [
    "JudgeSettings",
    "NORMAL_FORMAT_ATTEMPTS",
    "chat_completion",
    "local_constraint_body",
    "parallel_map",
    "parse_json_object",
    "resolve_judge_settings",
]
