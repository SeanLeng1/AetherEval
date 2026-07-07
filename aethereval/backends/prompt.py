
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
            except Exception as exc:  # noqa: BLE001
                _warn_chat_template_fallback(
                    f"apply_chat_template failed: {type(exc).__name__}: {exc}"
                )
        else:
            _warn_chat_template_fallback("missing apply_chat_template")

        lines = []
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    return str(prompt)


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


def render_prompt_with_chat_template(prompt: PromptType, tokenizer: Any) -> str:
    return _prompt_to_text(prompt, tokenizer)
