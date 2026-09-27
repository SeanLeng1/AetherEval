from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationInput, GenerationOutput
from aethereval.progress import Progress

from ..prompt import (
    _prompt_to_text,
    chat_template_kwargs_from_generation_config,
    count_text_tokens,
    count_token_ids,
    load_chat_tokenizer,
    prefilled_reasoning_prefix,
    validate_system_role_support as _validate_system_role_support,
)
from .service import SGLangService


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
    # No seed: SMG's /generate sampling params, the SGLang scheduler proto and
    # smg-grpc-servicer carry no per-request seed, so seeds apply to vLLM only.
    for key in ("regex", "json_schema", "ebnf", "structural_tag"):
        if gen_cfg.get(key) is not None:
            params[key] = gen_cfg[key]
    return params


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _parse_generate_response(
    output: Any,
) -> tuple[str, int | None, int | None, str | None]:
    """Parse one SMG /generate response: text, output_ids and meta_info."""

    if isinstance(output, list):
        if len(output) != 1:
            raise ValueError(
                f"SGLang gRPC returned an unexpected batch size: {len(output)}"
            )
        output = output[0]
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        meta_info = {}
    output_ids = output.get("output_ids")
    completion_tokens = (
        count_token_ids(output_ids)
        if output_ids is not None
        else _maybe_int(meta_info.get("completion_tokens"))
    )
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    return (
        str(output["text"]),
        _maybe_int(meta_info.get("prompt_tokens")),
        completion_tokens,
        finish_reason if isinstance(finish_reason, str) else None,
    )


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


def _run_service_generation(
    service: SGLangService,
    tokenizer: Any,
    payloads: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not payloads:
        return []

    sampling_params = _build_sampling_params(gen_cfg)
    chat_template_kwargs = chat_template_kwargs_from_generation_config(gen_cfg)
    request_items: list[dict[str, Any]] = []
    request_payloads: list[dict[str, Any]] = []
    prompt_token_counts: dict[int, int] = {}
    show_progress = bool(gen_cfg.get("_show_progress", True))
    with Progress(
        len(payloads), "sglang preparing prompts", "prompt", show_progress
    ) as progress:
        for item in payloads:
            rendered = _prompt_to_text(item["prompt"], tokenizer, chat_template_kwargs)
            for _ in range(int(item["num_generations"])):
                request_items.append(item)
                request_payloads.append(
                    {"text": rendered, "sampling_params": sampling_params}
                )
            progress.update()

    raw_outputs = service.request_many(
        "/generate",
        request_payloads,
        show_progress=show_progress,
        progress_desc="sglang generating",
        progress_unit="gen",
    )
    grouped_texts: dict[int, list[str]] = defaultdict(list)
    grouped_token_counts: dict[int, list[int | None]] = defaultdict(list)
    grouped_finish_reasons: dict[int, list[str | None]] = defaultdict(list)
    for item, request, output in zip(
        request_items, request_payloads, raw_outputs, strict=True
    ):
        item_idx = int(item["idx"])
        text, prompt_tokens, completion_tokens, finish_reason = (
            _parse_generate_response(output)
        )
        if item_idx not in prompt_token_counts:
            prompt_token_counts[item_idx] = (
                prompt_tokens
                if prompt_tokens is not None
                else count_text_tokens(request["text"], tokenizer)
            )
        grouped_texts[item_idx].append(
            prefilled_reasoning_prefix(request["text"]) + text
        )
        grouped_token_counts[item_idx].append(completion_tokens)
        grouped_finish_reasons[item_idx].append(finish_reason)

    results: list[dict[str, Any]] = []
    for item in payloads:
        item_idx = int(item["idx"])
        expected = int(item["num_generations"])
        texts = grouped_texts[item_idx]
        if len(texts) != expected:
            raise RuntimeError(
                f"SGLang returned {len(texts)} candidates for sample {item['sample_id']}; expected {expected}."
            )
        results.append(
            {
                "idx": item_idx,
                "sample_id": item["sample_id"],
                "prompt": item["prompt"],
                "generations": texts,
                "error": None,
                "meta": {
                    "prompt_token_count": prompt_token_counts[item_idx],
                    "response_token_counts": grouped_token_counts[item_idx],
                    "finish_reasons": grouped_finish_reasons[item_idx],
                },
            }
        )
    return results


class SGLangBackend:
    """SGLang generation backend.

    Ray manages every TP server and SMG routes every request, including when
    there is only one replica. Attached Ray worker nodes require no additional
    SGLang setup.
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

        self._tokenizer = None
        self._service: SGLangService | None = None

        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )

        self._service = SGLangService(
            model=self.model,
            dp_size=self.dp_size,
            tensor_parallel_size=self.tensor_parallel_size,
            model_kwargs=self.model_kwargs,
        )
        self._tokenizer = load_chat_tokenizer(self.model, self.model_kwargs)

    def validate_system_role_support(
        self, chat_template_kwargs: dict[str, Any]
    ) -> None:
        """Validate with an already-loaded tokenizer, including in Ray DP mode."""

        _validate_system_role_support(
            self._tokenizer,
            model=self.model,
            chat_template_kwargs=chat_template_kwargs,
        )

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

        assert self._service is not None
        output_dicts = _run_service_generation(
            self._service,
            self._tokenizer,
            payloads,
            gen_cfg,
        )

        return _outputs_from_dicts(output_dicts)

    def close(self) -> None:
        self._tokenizer = None
        if self._service is not None:
            self._service.close()
            self._service = None
        try:
            import ray
        except ImportError:
            return
        if ray.is_initialized():
            ray.shutdown()
