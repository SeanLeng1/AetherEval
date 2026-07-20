from typing import Any

from aethereval.backends.sglang.service import SGLangService


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


def _render_conversations(
    model_path: str,
    conversations: list[list[dict[str, str]]],
    *,
    max_length: int,
    trust_remote_code: bool,
) -> list[str]:
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
    rendered: list[str] = []
    for conversation in conversations:
        text = _format_conversation(tokenizer, conversation)
        ids = [
            int(token)
            for token in tokenizer.encode(text, add_special_tokens=True)
        ]
        if len(ids) > max_length:
            target_ids = ids[:max_length]
            text = tokenizer.decode(
                target_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            roundtrip_ids = [
                int(token)
                for token in tokenizer.encode(text, add_special_tokens=True)
            ]
            if roundtrip_ids != target_ids:
                raise RuntimeError(
                    "RM prompt cannot be losslessly truncated through the "
                    f"gRPC text API for model {model_path!r}"
                )
        rendered.append(text)
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
        max_length = int(options.get("max_length", 2048))
        if max_length < 1:
            raise ValueError(f"RM max_length must be >= 1, got {max_length}")
        trust_remote_code = bool(options.get("trust_remote_code", True))
        dtype = options.get("dtype", "auto")
        extra_sglang_args = options.get("sglang_args", {})
        if not isinstance(extra_sglang_args, dict):
            raise ValueError("RM sglang_args must be a mapping/object")

        results: dict[str, list[float]] = {}
        for model_path in unique_paths:
            rendered_inputs = _render_conversations(
                model_path,
                conversations,
                max_length=max_length,
                trust_remote_code=trust_remote_code,
            )
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
