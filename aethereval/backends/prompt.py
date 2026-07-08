import warnings
from typing import Any

from aethereval.core.types import PromptType


_CHAT_TEMPLATE_FALLBACK_WARNED = False


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


def _prompt_to_text(prompt: PromptType, tokenizer: Any) -> str:
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    if isinstance(prompt, list):
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
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


def render_prompt_with_chat_template(prompt: PromptType, tokenizer: Any) -> str:
    return _prompt_to_text(prompt, tokenizer)
