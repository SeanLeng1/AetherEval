from __future__ import annotations

from collections import defaultdict
import warnings
from typing import Any

from .types import GenerationInput, GenerationOutput, PromptType


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
                _warn_chat_template_fallback(f"apply_chat_template failed: {type(exc).__name__}: {exc}")
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


def _build_sampling_params(vllm_module: Any, gen_cfg: dict[str, Any], n: int) -> Any:
    kwargs: dict[str, Any] = {
        "n": int(n),
        "max_tokens": int(gen_cfg.get("max_new_tokens", 256)),
        "temperature": float(gen_cfg.get("temperature", 0.0)),
        "top_p": float(gen_cfg.get("top_p", 1.0)),
    }
    if gen_cfg.get("top_k") is not None:
        kwargs["top_k"] = int(gen_cfg["top_k"])
    if gen_cfg.get("min_p") is not None:
        kwargs["min_p"] = float(gen_cfg["min_p"])
    if gen_cfg.get("stop") is not None:
        kwargs["stop"] = gen_cfg["stop"]
    if gen_cfg.get("seed") is not None:
        kwargs["seed"] = int(gen_cfg["seed"])
    return vllm_module.SamplingParams(**kwargs)


def _run_generation(
    llm: Any,
    tokenizer: Any,
    vllm_module: Any,
    payloads: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
    show_progress: bool,
) -> list[dict[str, Any]]:
    if not payloads:
        return []

    bucketed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payloads:
        bucketed[int(item["num_generations"])].append(item)

    result_by_index: dict[int, dict[str, Any]] = {}

    for n, items in bucketed.items():
        sampling_params = _build_sampling_params(vllm_module, gen_cfg, n=n)
        prompts = [_prompt_to_text(x["prompt"], tokenizer) for x in items]
        try:
            outputs = llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=show_progress,
            )
            for item, output in zip(items, outputs):
                texts = [candidate.text for candidate in output.outputs]
                if len(texts) < n:
                    texts.extend([""] * (n - len(texts)))
                result_by_index[item["idx"]] = {
                    "idx": item["idx"],
                    "sample_id": item["sample_id"],
                    "prompt": item["prompt"],
                    "generations": texts[:n],
                    "error": None,
                }
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            for item in items:
                result_by_index[item["idx"]] = {
                    "idx": item["idx"],
                    "sample_id": item["sample_id"],
                    "prompt": item["prompt"],
                    "generations": [""] * n,
                    "error": err,
                }

    return [result_by_index[i] for i in sorted(result_by_index)]


class VLLMBackend:
    """
    Generative-only vLLM backend.

    - dp_size = 1: single process vLLM.
    - dp_size > 1: Ray data-parallel workers.
    """

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
        self.model_kwargs = model_kwargs or {}

        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )

        self._vllm = None
        self._ray = None
        self._llm = None
        self._tokenizer = None
        self._workers: list[Any] = []

        if self.dp_size == 1:
            self._init_single()
        else:
            self._init_ray()

    def _import_vllm(self) -> Any:
        try:
            import vllm
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Install dependencies first (`pip install -e .`)."
            ) from exc
        self._vllm = vllm
        return vllm

    def _init_single(self) -> None:
        vllm = self._import_vllm()
        llm_args = {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": False,
        }
        llm_args.update(self.model_kwargs)
        self._llm = vllm.LLM(**llm_args)
        self._tokenizer = self._llm.get_tokenizer()

    def _init_ray(self) -> None:
        self._import_vllm()
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError(
                "ray is not installed. Install dependencies first (`pip install -e .`)."
            ) from exc

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._ray = ray

        llm_args = {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": False,
        }
        llm_args.update(self.model_kwargs)

        @ray.remote(num_gpus=self.tensor_parallel_size)
        class _Worker:  # noqa: D401
            def __init__(self, _llm_args: dict[str, Any]) -> None:
                import vllm as _vllm

                self._vllm = _vllm
                self._llm = _vllm.LLM(**_llm_args)
                self._tokenizer = self._llm.get_tokenizer()

            def generate(
                self,
                payloads: list[dict[str, Any]],
                gen_cfg: dict[str, Any],
            ) -> list[dict[str, Any]]:
                return _run_generation(
                    llm=self._llm,
                    tokenizer=self._tokenizer,
                    vllm_module=self._vllm,
                    payloads=payloads,
                    gen_cfg=gen_cfg,
                    show_progress=True,
                )

        self._workers = [_Worker.remote(llm_args) for _ in range(self.dp_size)]

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

        if self.dp_size == 1:
            assert self._llm is not None and self._vllm is not None
            output_dicts = _run_generation(
                llm=self._llm,
                tokenizer=self._tokenizer,
                vllm_module=self._vllm,
                payloads=payloads,
                gen_cfg=gen_cfg,
                show_progress=True,
            )
        else:
            assert self._ray is not None
            worker_payloads = [[] for _ in self._workers]
            for idx, payload in enumerate(payloads):
                worker_payloads[idx % len(self._workers)].append(payload)

            refs = []
            for worker, worker_items in zip(self._workers, worker_payloads):
                refs.append(worker.generate.remote(worker_items, gen_cfg))
            nested = self._ray.get(refs)
            output_dicts = [item for sublist in nested for item in sublist]
            output_dicts.sort(key=lambda x: x["idx"])

        return [
            GenerationOutput(
                sample_id=item["sample_id"],
                prompt=item["prompt"],
                generations=item["generations"],
                error=item["error"],
            )
            for item in output_dicts
        ]

    def close(self) -> None:
        self._llm = None
        self._tokenizer = None
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
        self._workers = []
