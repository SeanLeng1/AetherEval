from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class StandaloneRewardModelBackend:
    """Reward-model scoring runtime that does not load a candidate LLM engine."""

    name = "standalone-reward-model"

    def __init__(self, devices: list[str]) -> None:
        normalized = [str(device).strip() for device in devices if str(device).strip()]
        if not normalized:
            raise ValueError("Standalone reward-model scoring requires a device")
        self.devices = normalized

    def score_reward_models(
        self,
        model_paths: list[str],
        conversations: list[list[dict[str, str]]],
        scorer_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        return score_conversations_sharded(
            model_paths=model_paths,
            conversations=conversations,
            devices=self.devices,
            scorer_kwargs=dict(scorer_kwargs or {}),
        )

    def close(self) -> None:
        # score_conversations_sharded owns and releases each model process.
        return None


def _torch_dtype(dtype_name: str | None, torch: Any) -> Any:
    if dtype_name is None or dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    normalized = dtype_name.strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported RM dtype: {dtype_name}")
    return mapping[normalized]


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
        content = message["content"]
        text += f"{role}: {content}\n"
    return text


def _score_tensor_from_output(output: Any) -> Any:
    if hasattr(output, "end_scores") and output.end_scores is not None:
        return output.end_scores.squeeze(-1)
    if hasattr(output, "logits") and output.logits is not None:
        logits = output.logits
        if logits.dim() == 3:
            return logits[:, -1, :].squeeze(-1)
        return logits.squeeze(-1)
    if isinstance(output, (tuple, list)) and output:
        tensor = output[0]
        if tensor.dim() == 3:
            return tensor[:, -1, :].squeeze(-1)
        return tensor.squeeze(-1)
    raise ValueError("Reward model output does not contain end_scores or logits")


def _load_reward_model_class(model_path: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for RM-based metrics. Install transformers first."
        ) from exc

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if config.model_type == "qwen2":
        try:
            import torch
            import torch.nn as nn
            from transformers import Qwen2Model, Qwen2PreTrainedModel
            from transformers.utils import ModelOutput
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2 reward models require torch and a transformers build with Qwen2 support."
            ) from exc

        @dataclass
        class ScoreModelOutput(ModelOutput):
            scores: Any | None = None
            end_scores: Any | None = None
            last_hidden_state: Any | None = None
            end_last_hidden_state: Any | None = None
            end_index: Any | None = None

        class Qwen2RewardModel(Qwen2PreTrainedModel):
            supports_gradient_checkpointing = True

            def __init__(self, config: Any) -> None:
                super().__init__(config)
                setattr(self, self.base_model_prefix, Qwen2Model(config))
                self.score_head = nn.Linear(config.hidden_size, 1, bias=False)
                self.post_init()

            def forward(
                self,
                input_ids: Any | None = None,
                attention_mask: Any | None = None,
                **kwargs: Any,
            ) -> Any:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    **kwargs,
                )
                last_hidden_state = outputs.hidden_states[-1]
                scores = self.score_head(last_hidden_state).float()
                batch_size, seq_len, _ = last_hidden_state.size()

                if attention_mask is None:
                    if batch_size > 1:
                        raise ValueError(
                            "'attention_mask' is required when batch size > 1."
                        )
                    attention_mask = last_hidden_state.new_ones(
                        batch_size, seq_len, dtype=torch.bool
                    )

                end_index = torch.cat([mask.nonzero()[-1] for mask in attention_mask])
                gather_hidden_index = (
                    end_index.to(last_hidden_state.device)
                    .unsqueeze(dim=1)
                    .unsqueeze(dim=2)
                    .expand(-1, -1, last_hidden_state.size(-1))
                )
                gather_score_index = (
                    end_index.to(scores.device)
                    .unsqueeze(dim=1)
                    .unsqueeze(dim=2)
                    .expand(-1, -1, scores.size(-1))
                )
                end_last_hidden_state = torch.gather(
                    last_hidden_state,
                    dim=1,
                    index=gather_hidden_index,
                ).squeeze(dim=1)
                end_scores = torch.gather(
                    scores,
                    dim=1,
                    index=gather_score_index,
                ).squeeze(dim=1)

                return ScoreModelOutput(
                    scores=scores,
                    end_scores=end_scores,
                    last_hidden_state=last_hidden_state,
                    end_last_hidden_state=end_last_hidden_state,
                    end_index=end_index,
                )

        return Qwen2RewardModel

    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification


class RewardModelScorer:
    def __init__(
        self,
        *,
        model_path: str,
        batch_size: int = 1,
        max_length: int = 2048,
        device: str | None = None,
        dtype: str | None = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for RM-based metrics."
            ) from exc

        self._torch = torch
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_cls = _load_reward_model_class(model_path, trust_remote_code)
        torch_dtype = _torch_dtype(dtype, torch)
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    def score(self, conversations: list[list[dict[str, str]]]) -> list[float]:
        scores: list[float] = []
        starts = range(0, len(conversations), self.batch_size)
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                starts,
                desc=f"RM scoring {self.model_path}",
                unit="batch",
                dynamic_ncols=True,
            )
        except ImportError:
            iterator = starts

        with self._torch.inference_mode():
            for start in iterator:
                batch = conversations[start : start + self.batch_size]
                texts = [
                    _format_conversation(self.tokenizer, conversation)
                    for conversation in batch
                ]
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                model_inputs = {
                    "input_ids": inputs["input_ids"].to(self.device),
                    "attention_mask": inputs["attention_mask"].to(self.device),
                }
                output = self.model(**model_inputs)
                tensor = _score_tensor_from_output(output)
                scores.extend(float(value) for value in tensor.detach().cpu().tolist())
        return scores

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _score_shard_on_device(
    args: tuple[list[str], list[list[dict[str, str]]], str, dict[str, Any]],
) -> dict[str, list[float]]:
    # Module-level so multiprocessing spawn workers can pickle it.
    model_paths, conversations, device, scorer_kwargs = args
    results: dict[str, list[float]] = {}
    for model_path in dict.fromkeys(model_paths):
        scorer = RewardModelScorer(
            model_path=model_path, device=device, **scorer_kwargs
        )
        try:
            results[model_path] = scorer.score(conversations)
        finally:
            scorer.close()
    return results


def score_conversations_sharded(
    *,
    model_paths: list[str],
    conversations: list[list[dict[str, str]]],
    devices: list[str],
    scorer_kwargs: dict[str, Any],
) -> dict[str, list[float]]:
    """Score every conversation with each model, data-parallel across devices.

    Contiguous shards, one spawn worker per device; each worker keeps one reward
    model resident at a time (models scored sequentially within a shard).
    """
    unique_paths = list(dict.fromkeys(model_paths))
    if not unique_paths:
        raise ValueError("model_paths must not be empty")
    if not devices:
        raise ValueError("devices must not be empty")
    if not conversations:
        return {path: [] for path in unique_paths}

    num_workers = min(len(devices), len(conversations))
    if num_workers == 1:
        return _score_shard_on_device(
            (unique_paths, conversations, devices[0], scorer_kwargs)
        )

    base, extra = divmod(len(conversations), num_workers)
    shards: list[list[list[dict[str, str]]]] = []
    start = 0
    for worker_index in range(num_workers):
        size = base + (1 if worker_index < extra else 0)
        shards.append(conversations[start : start + size])
        start += size

    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=num_workers) as pool:
        shard_results = pool.map(
            _score_shard_on_device,
            [
                (unique_paths, shard, devices[worker_index], scorer_kwargs)
                for worker_index, shard in enumerate(shards)
            ],
        )

    merged: dict[str, list[float]] = {path: [] for path in unique_paths}
    for shard_result in shard_results:
        for path in unique_paths:
            merged[path].extend(shard_result[path])
    for path in unique_paths:
        if len(merged[path]) != len(conversations):
            raise ValueError(
                f"sharded scoring returned {len(merged[path])} scores for {path}, "
                f"expected {len(conversations)}"
            )
    return merged
