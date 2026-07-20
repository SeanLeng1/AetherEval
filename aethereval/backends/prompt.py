import warnings
from typing import Any

from aethereval.core.types import PromptType


_CHAT_TEMPLATE_FALLBACK_WARNED = False
_SYSTEM_ROLE_PROBE = "AETHEREVAL_SYSTEM_ROLE_SUPPORT_PROBE_7f34c8"


def _warn_chat_template_fallback(reason: str | None = None) -> None:
    global _CHAT_TEMPLATE_FALLBACK_WARNED
    if _CHAT_TEMPLATE_FALLBACK_WARNED:
        return
    suffix = f" ({reason})" if reason else ""
    warnings.warn(
        "Tokenizer has no usable chat template; falling back to plain 'role: content' prompt formatting."
        + suffix,
        RuntimeWarning,
        stacklevel=2,
    )
    _CHAT_TEMPLATE_FALLBACK_WARNED = True


def chat_template_kwargs_from_generation_config(
    gen_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    if not gen_cfg or gen_cfg.get("enable_thinking") is None:
        return {}
    value = gen_cfg["enable_thinking"]
    if not isinstance(value, bool):
        raise ValueError(
            f"enable_thinking must be true or false when provided, got {value!r}"
        )
    return {"enable_thinking": value}


def validate_system_role_support(
    tokenizer: Any,
    *,
    model: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> None:
    """Fail clearly if a judge tokenizer cannot preserve a system message."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError(
            f"Judge model {model!r} has no usable chat template, so AetherEval "
            "cannot send the system-role messages required by this benchmark."
        )

    messages = [
        {"role": "system", "content": _SYSTEM_ROLE_PROBE},
        {"role": "user", "content": "Reply with OK."},
    ]
    try:
        rendered = apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    except Exception as exc:
        raise ValueError(
            f"Judge model {model!r} chat template does not support system-role "
            "messages required by this benchmark. Choose a compatible judge "
            f"model or chat template. Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(rendered, str) or _SYSTEM_ROLE_PROBE not in rendered:
        raise ValueError(
            f"Judge model {model!r} chat template does not preserve system-role "
            "content required by this benchmark. Choose a compatible judge "
            "model or chat template."
        )


def _prompt_to_text(
    prompt: PromptType,
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    if isinstance(prompt, list):
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(chat_template_kwargs or {}),
                )
            except ValueError as exc:
                _warn_chat_template_fallback(
                    f"apply_chat_template failed: {type(exc).__name__}: {exc}"
                )
        else:
            _warn_chat_template_fallback("missing apply_chat_template")

        lines = []
        for idx, message in enumerate(prompt):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Invalid chat message at index {idx}: expected dict, got {type(message).__name__}"
                )
            role = message["role"]
            content = message["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


def load_chat_tokenizer(
    model: str,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for prompt inspection. Install dependencies first."
        ) from exc

    kwargs = model_kwargs or {}
    tokenizer_name = str(kwargs.get("tokenizer", model))
    tokenizer_args: dict[str, Any] = {}
    for key in ("trust_remote_code", "revision", "tokenizer_revision"):
        if key in kwargs and kwargs[key] is not None:
            tokenizer_args[key] = kwargs[key]

    return AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_args)


def count_token_ids(token_ids: Any) -> int:
    if token_ids is None:
        raise ValueError("token_ids must not be None")
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return len(token_ids)


def count_text_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for token counting.")

    if hasattr(tokenizer, "encode"):
        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(text)
        return count_token_ids(token_ids)

    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        return count_token_ids(encoded["input_ids"])
    input_ids = getattr(encoded, "input_ids")
    return count_token_ids(input_ids)


def render_prompt_with_chat_template(
    prompt: PromptType,
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    return _prompt_to_text(prompt, tokenizer, chat_template_kwargs)
