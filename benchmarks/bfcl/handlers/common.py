"""Shared transport for AetherEval's BFCL prompt-mode handlers."""

import os
import threading
import time
from types import SimpleNamespace

import requests

from ..errors import is_context_length_error


_REQUEST_STATE = threading.local()
_NATIVE_GENERATE_TIMEOUT = (30, 1800)
# SGLang reserves a few positions for request bookkeeping/special tokens.
_SERVER_CONTEXT_MARGIN = 8


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _single_generate_result(result):
    while isinstance(result, list):
        if len(result) != 1:
            raise ValueError(
                "BFCL native generation returned an unexpected batch size: "
                f"{len(result)}"
            )
        result = result[0]
    if not isinstance(result, dict):
        raise TypeError(
            "BFCL native generation returned an unsupported response type: "
            f"{type(result).__name__}"
        )
    return result


def _generate_text(result: dict) -> str:
    for key in ("text", "output_text", "generated_text"):
        if key in result:
            return str(result[key])
    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs:
        return _generate_text(_single_generate_result(outputs[0]))
    raise ValueError("BFCL native generation response contains no generated text.")


def _token_count(result: dict, keys: tuple[str, ...]) -> int | None:
    for source in (result, result.get("meta_info")):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _request_session() -> requests.Session:
    session = getattr(_REQUEST_STATE, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _REQUEST_STATE.session = session
    return session


def query_rendered_prompt(
    handler,
    formatted_prompt: str,
    *,
    skip_special_tokens: bool | None,
):
    """Generate one already-rendered prompt through the configured BFCL backend."""
    input_token_count = len(handler.tokenizer.tokenize(formatted_prompt))
    max_tokens = _env_int("AETHEREVAL_BFCL_MAX_TOKENS", 4096)
    if max_tokens is None or max_tokens <= 0:
        raise ValueError("AETHEREVAL_BFCL_MAX_TOKENS must be positive.")

    max_context_length = _env_int(
        "AETHEREVAL_BFCL_MAX_CONTEXT_LENGTH",
        handler.max_context_length,
    )
    if max_context_length is None or max_context_length <= 0:
        raise ValueError("BFCL max context length must be positive.")

    available_tokens = max_context_length - input_token_count - _SERVER_CONTEXT_MARGIN
    if available_tokens <= 0:
        raise ValueError(
            "BFCL prompt exceeds max context length: "
            f"input_tokens={input_token_count}, "
            f"max_context_length={max_context_length}."
        )
    leftover_tokens_count = min(max_tokens, available_tokens)

    extra_body = {
        "repetition_penalty": _env_float("AETHEREVAL_BFCL_REPETITION_PENALTY", 1.0),
        "top_p": _env_float("AETHEREVAL_BFCL_TOP_P", 1.0),
        "top_k": _env_int("AETHEREVAL_BFCL_TOP_K", -1),
    }
    seed = _env_int("AETHEREVAL_BFCL_SEED")
    if seed is not None:
        extra_body["seed"] = seed
    if hasattr(handler, "stop_token_ids"):
        extra_body["stop_token_ids"] = handler.stop_token_ids
    if skip_special_tokens is not None:
        extra_body["skip_special_tokens"] = skip_special_tokens

    start_time = time.time()
    generate_url = os.getenv("AETHEREVAL_BFCL_GENERATE_URL")
    if generate_url:
        sampling_params = {
            "max_new_tokens": leftover_tokens_count,
            "temperature": handler.temperature,
            **extra_body,
        }
        if sampling_params.get("top_k") == -1:
            sampling_params.pop("top_k")
        result = post_native_generate(
            generate_url,
            {
                "model": handler.model_path_or_id,
                "text": formatted_prompt,
                "sampling_params": sampling_params,
            },
        )
        generated_text = _generate_text(result)
        prompt_tokens = _token_count(
            result,
            ("prompt_tokens", "input_tokens", "input_token_count"),
        )
        completion_tokens = _token_count(
            result,
            (
                "completion_tokens",
                "output_tokens",
                "output_token_count",
                "num_output_tokens",
            ),
        )
        api_response = SimpleNamespace(
            choices=[SimpleNamespace(text=generated_text)],
            usage=SimpleNamespace(
                prompt_tokens=(
                    input_token_count if prompt_tokens is None else prompt_tokens
                ),
                completion_tokens=(
                    len(handler.tokenizer.tokenize(generated_text))
                    if completion_tokens is None
                    else completion_tokens
                ),
            ),
        )
    else:
        api_response = handler.client.completions.create(
            model=handler.model_path_or_id,
            temperature=handler.temperature,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            extra_body=extra_body,
            timeout=_NATIVE_GENERATE_TIMEOUT[1],
        )
    return api_response, time.time() - start_time


def post_native_generate(url: str, payload: dict) -> dict:
    """Issue one native request, retrying only transient transport/server failures."""
    for attempt in range(3):
        try:
            response = _request_session().post(
                url,
                json=payload,
                timeout=_NATIVE_GENERATE_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        else:
            if response.ok:
                try:
                    response_body = response.json()
                except ValueError as exc:
                    if attempt == 2:
                        raise RuntimeError(
                            "BFCL native generation returned malformed JSON after "
                            f"3 attempts: {response.text[:500]!r}"
                        ) from exc
                    time.sleep(0.5 * (2**attempt))
                    continue
                return _single_generate_result(response_body)
            if is_context_length_error(response.text):
                raise RuntimeError(
                    "BFCL native generation rejected an overlength prompt "
                    f"(not retried), HTTP {response.status_code}: {response.text}"
                )
            if response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(
                    "BFCL native generation failed with HTTP "
                    f"{response.status_code}: {response.text}"
                )
            if attempt == 2:
                raise RuntimeError(
                    "BFCL native generation failed with HTTP "
                    f"{response.status_code}: {response.text}"
                )
        time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")
