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
    if gen_cfg.get("seed") is not None:
        params["seed"] = int(gen_cfg["seed"])
    for key in ("regex", "json_schema", "ebnf", "structural_tag"):
        if gen_cfg.get(key) is not None:
            params[key] = gen_cfg[key]
    return params


def _extract_text(output: Any) -> str:
    if isinstance(output, list):
        if len(output) != 1:
            raise ValueError(
                f"SGLang gRPC returned an unexpected batch size: {len(output)}"
            )
        return _extract_text(output[0])
    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        for key in ("text", "output_text", "generated_text"):
            if key in output:
                return str(output[key])
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"])
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return _extract_text(outputs[0])

    text = getattr(output, "text", None)
    if text is not None:
        return str(text)

    raise TypeError(f"Unsupported SGLang output type: {type(output).__name__}")


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _dict_first_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _maybe_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _maybe_count_token_ids(token_ids: Any) -> int | None:
    if token_ids is None:
        return None
    try:
        return count_token_ids(token_ids)
    except TypeError:
        return None


def _extract_output_token_count(output: Any) -> int | None:
    if isinstance(output, list):
        if len(output) != 1:
            raise ValueError(
                f"SGLang gRPC returned an unexpected batch size: {len(output)}"
            )
        return _extract_output_token_count(output[0])
    if isinstance(output, str):
        return None

    if isinstance(output, dict):
        for key in ("output_token_ids", "output_ids", "token_ids"):
            count = _maybe_count_token_ids(output.get(key))
            if count is not None:
                return count
        count = _dict_first_int(
            output,
            (
                "output_token_count",
                "completion_token_count",
                "num_output_tokens",
                "num_completion_tokens",
                "completion_tokens",
            ),
        )
        if count is not None:
            return count
        meta_info = output.get("meta_info")
        if isinstance(meta_info, dict):
            count = _dict_first_int(
                meta_info,
                (
                    "output_token_count",
                    "completion_token_count",
                    "num_output_tokens",
                    "num_completion_tokens",
                    "completion_tokens",
                ),
            )
            if count is not None:
                return count
        choices = output.get("choices")
        if isinstance(choices, list) and choices:
            return _extract_output_token_count(choices[0])
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return _extract_output_token_count(outputs[0])
        return None

    for attr in ("output_token_ids", "output_ids", "token_ids"):
        count = _maybe_count_token_ids(getattr(output, attr, None))
        if count is not None:
            return count
    for attr in (
        "output_token_count",
        "completion_token_count",
        "num_output_tokens",
        "num_completion_tokens",
        "completion_tokens",
    ):
        count = _maybe_int(getattr(output, attr, None))
        if count is not None:
            return count
    meta_info = getattr(output, "meta_info", None)
    if isinstance(meta_info, dict):
        return _dict_first_int(
            meta_info,
            (
                "output_token_count",
                "completion_token_count",
                "num_output_tokens",
                "num_completion_tokens",
                "completion_tokens",
            ),
        )
    return None


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


def _extract_prompt_token_count(output: Any) -> int | None:
    if isinstance(output, list) and len(output) == 1:
        return _extract_prompt_token_count(output[0])
    if isinstance(output, dict):
        for values in (output.get("meta_info"), output.get("usage"), output):
            if isinstance(values, dict):
                count = _dict_first_int(
                    values, ("prompt_tokens", "prompt_token_count", "input_token_count")
                )
                if count is not None:
                    return count
    return None


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
    for item, request, output in zip(
        request_items, request_payloads, raw_outputs, strict=True
    ):
        item_idx = int(item["idx"])
        if item_idx not in prompt_token_counts:
            count = _extract_prompt_token_count(output)
            prompt_token_counts[item_idx] = (
                count
                if count is not None
                else count_text_tokens(request["text"], tokenizer)
            )
        grouped_texts[item_idx].append(_extract_text(output))
        grouped_token_counts[item_idx].append(_extract_output_token_count(output))

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
