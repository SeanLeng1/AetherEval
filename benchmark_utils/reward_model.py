import json
from typing import Any

from aethereval.backends.sglang.service import SGLangService

# GPT-2 classification admits the full native context without generation.
GPT2_INPUT_LIMIT = 1024
SAFERLHF_INPUT_LIMIT = 2048
# Chat inputs are tokenized in batches of this many conversations: a batched
# fast-tokenizer call returns the same ids as per-text calls, runs in Rust
# threads, and the chunks bound the memory of the rendered texts.
ENCODE_CHUNK_SIZE = 1024


def gpt2_reward_input(conversation, tokenizer):
    roles = {"user": "Human", "assistant": "Assistant", "system": "System"}
    query = "".join(f"\n\n{roles[m['role']]}: {m['content']}" for m in conversation[:-1]).rstrip() + " \n\nAssistant:"
    encoded = tokenizer(query, conversation[-1]["content"].strip(), truncation=True, max_length=GPT2_INPUT_LIMIT)
    return encoded["input_ids"]


def _format_conversation(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=True,
        )

    text = ""
    for message in messages:
        role = str(message["role"]).upper()
        text += f"{role}: {message['content']}\n"
    return text


def saferlhf_reward_input(conversation, tokenizer):
    text = _format_conversation(tokenizer, conversation)
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) > SAFERLHF_INPUT_LIMIT:
        ids = ids[:SAFERLHF_INPUT_LIMIT]
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=True) != ids:
            raise RuntimeError("SafeRLHF input cannot be losslessly truncated through the text API")
    return text


def _load_tokenizer(model_path: str, *, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to render reward-model conversations"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.truncation_side = "right"
    return tokenizer


def _tokenizer_fingerprint(tokenizer: Any) -> str | None:
    """Everything that shapes rendered inputs; None never shares a render."""

    cls = type(tokenizer)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or cls.__module__.startswith("transformers_modules"):
        return None
    return json.dumps(
        [
            cls.__module__,
            cls.__qualname__,
            backend.to_str(),
            tokenizer.chat_template,
            tokenizer.special_tokens_map,
            getattr(tokenizer, "split_special_tokens", None),
        ],
        sort_keys=True,
        default=str,
    )


def _render_conversations(
    tokenizer: Any,
    conversations: list[list[dict[str, str]]],
    *,
    reward_format: str = "chat",
    max_length: int | None = None,
) -> list[str | list[int]]:
    if reward_format == "gpt2":
        return [gpt2_reward_input(c, tokenizer) for c in conversations]
    if reward_format != "chat":
        raise ValueError(f"Unknown reward input format: {reward_format}")
    if max_length is None:
        return [saferlhf_reward_input(c, tokenizer) for c in conversations]

    rendered: list[str | list[int]] = []
    for start in range(0, len(conversations), ENCODE_CHUNK_SIZE):
        texts = [
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False,
            )
            for conversation in conversations[start : start + ENCODE_CHUNK_SIZE]
        ]
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        rendered.extend(encoded["input_ids"])
    return rendered


def _extract_scalar_embedding(response: Any) -> float:
    if not isinstance(response, dict):
        raise ValueError(
            "SGLang reward model returned a non-object response: "
            f"{type(response).__name__}"
        )
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("SGLang reward model returned invalid embedding data")
    item = data[0]
    if not isinstance(item, dict):
        raise ValueError("SGLang reward model returned invalid embedding item")
    embedding = item.get("embedding")
    if not isinstance(embedding, list) or len(embedding) != 1:
        raise ValueError(
            "Safe-alignment reward models must return exactly one raw score"
        )
    return float(embedding[0])


class SGLangRewardModelBackend:
    """Score converted sequence-classification checkpoints with SGLang.

    RM and CM are loaded sequentially. Each model uses the complete requested
    DP x TP GPU budget, and SMG dynamically routes conversations across all
    replicas on the attached Ray cluster.
    """

    name = "sglang-reward-model"

    def __init__(
        self,
        *,
        dp_size: int,
        tensor_parallel_size: int,
    ) -> None:
        self.dp_size = int(dp_size)
        self.tensor_parallel_size = int(tensor_parallel_size)
        if self.dp_size < 1:
            raise ValueError(f"RM dp_size must be >= 1, got {self.dp_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                "RM tensor_parallel_size must be >= 1, "
                f"got {self.tensor_parallel_size}"
            )

    def score_reward_models(
        self,
        model_paths: list[str],
        conversations: list[list[dict[str, str]]],
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        unique_paths = list(dict.fromkeys(model_paths))
        if not unique_paths:
            raise ValueError("model_paths must not be empty")
        if not conversations:
            return {path: [] for path in unique_paths}

        options = dict(scorer_kwargs or {})
        trust_remote_code = bool(options.get("trust_remote_code", True))
        dtype = options.get("dtype", "auto")
        extra_sglang_args = options.get("sglang_args", {})
        if not isinstance(extra_sglang_args, dict):
            raise ValueError("RM sglang_args must be a mapping/object")

        results: dict[str, list[float]] = {}
        # RM and CM checkpoints often share one tokenizer; render their inputs once.
        rendered_by_tokenizer: dict[str, list[str | list[int]]] = {}
        for model_path in unique_paths:
            tokenizer = _load_tokenizer(model_path, trust_remote_code=trust_remote_code)
            fingerprint = _tokenizer_fingerprint(tokenizer)
            rendered_inputs = (
                rendered_by_tokenizer.get(fingerprint) if fingerprint else None
            )
            if rendered_inputs is None:
                rendered_inputs = _render_conversations(
                    tokenizer,
                    conversations,
                    reward_format=options.get("reward_format", "chat"),
                    max_length=options.get("max_length"),
                )
                if fingerprint:
                    rendered_by_tokenizer[fingerprint] = rendered_inputs
            model_kwargs = dict(extra_sglang_args)
            # Sequence-classification scoring is prefill-only. Capturing the
            # large default prefill CUDA-graph matrix adds minutes to every RM
            # and CM startup without changing the forward result.
            model_kwargs.setdefault("cuda_graph_backend_decode", "disabled")
            model_kwargs.setdefault("cuda_graph_backend_prefill", "disabled")
            model_kwargs.setdefault("is_embedding", True)
            if trust_remote_code:
                model_kwargs.setdefault("trust_remote_code", True)
            if dtype is not None and str(dtype).lower() != "auto":
                model_kwargs.setdefault("dtype", str(dtype))

            service = SGLangService(
                model=model_path,
                dp_size=self.dp_size,
                tensor_parallel_size=self.tensor_parallel_size,
                model_kwargs=model_kwargs,
            )
            try:
                responses = service.request_many(
                    "/v1/embeddings",
                    [
                        {
                            "model": model_path,
                            "input": text,
                        }
                        for text in rendered_inputs
                    ],
                    show_progress=True,
                    progress_desc=f"RM scoring {model_path}",
                    progress_unit="sample",
                )
                results[model_path] = [
                    _extract_scalar_embedding(response) for response in responses
                ]
            finally:
                service.close()
        return results

    def close(self) -> None:
        try:
            import ray
        except ImportError:
            return
        if ray.is_initialized():
            ray.shutdown()


__all__ = ["SGLangRewardModelBackend"]
