
from collections import defaultdict
from typing import Any

from aethereval.core.types import GenerationInput, GenerationOutput

from ..prompt import _prompt_to_text, load_chat_tokenizer


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
    return params


def _extract_text(output: Any) -> str:
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

    return str(output)


def _normalize_outputs(outputs: Any) -> list[Any]:
    if isinstance(outputs, dict):
        return [outputs]
    if isinstance(outputs, list):
        return outputs
    try:
        return list(outputs)
    except TypeError:
        return [outputs]


def _resolve_generation_batch_size(raw_value: Any) -> int:
    batch_size = int(raw_value if raw_value is not None else 128)
    if batch_size < 1:
        raise ValueError(f"generation_batch_size must be >= 1, got {batch_size}")
    return batch_size


def _make_progress_bar(total: int, desc: str, enabled: bool) -> Any:
    if not enabled or total <= 0:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None
    return tqdm(total=total, desc=desc, unit="gen", dynamic_ncols=True)


def _split_payloads(
    payloads: list[dict[str, Any]],
    num_workers: int,
) -> list[list[dict[str, Any]]]:
    worker_payloads: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    for idx, payload in enumerate(payloads):
        worker_payloads[idx % num_workers].append(payload)
    return worker_payloads


def _outputs_from_dicts(output_dicts: list[dict[str, Any]]) -> list[GenerationOutput]:
    return [
        GenerationOutput(
            sample_id=item["sample_id"],
            prompt=item["prompt"],
            generations=item["generations"],
            error=item["error"],
        )
        for item in output_dicts
    ]


def _run_generation(
    engine: Any,
    tokenizer: Any,
    payloads: list[dict[str, Any]],
    gen_cfg: dict[str, Any],
    batch_size: int = 128,
    show_progress: bool = True,
    progress_desc: str = "sglang generating",
) -> list[dict[str, Any]]:
    if not payloads:
        return []

    batch_size = _resolve_generation_batch_size(batch_size)
    bucketed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payloads:
        bucketed[int(item["num_generations"])].append(item)

    result_by_index: dict[int, dict[str, Any]] = {}
    sampling_params = _build_sampling_params(gen_cfg)
    progress_total = sum(int(item["num_generations"]) for item in payloads)
    progress_bar = _make_progress_bar(progress_total, progress_desc, show_progress)

    try:
        for n, items in bucketed.items():
            prompts: list[str] = []
            request_payloads: list[dict[str, Any]] = []
            for item in items:
                rendered = _prompt_to_text(item["prompt"], tokenizer)
                for _ in range(n):
                    prompts.append(rendered)
                    request_payloads.append(item)

            grouped_texts: dict[int, list[str]] = defaultdict(list)
            try:
                for start in range(0, len(prompts), batch_size):
                    end = start + batch_size
                    batch_prompts = prompts[start:end]
                    batch_payloads = request_payloads[start:end]
                    outputs = _normalize_outputs(
                        engine.generate(batch_prompts, sampling_params)
                    )
                    for item, output in zip(batch_payloads, outputs):
                        grouped_texts[int(item["idx"])].append(_extract_text(output))
                    if progress_bar is not None:
                        progress_bar.update(len(batch_payloads))

                for item in items:
                    texts = grouped_texts[int(item["idx"])]
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
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return [result_by_index[i] for i in sorted(result_by_index)]


class SGLangBackend:
    """SGLang offline backend.

    - dp_size = 1: single SGLang Engine.
    - dp_size > 1: Ray data-parallel SGLang Engine workers.
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
        self.generation_batch_size = _resolve_generation_batch_size(
            self.model_kwargs.pop("generation_batch_size", None)
        )

        self._sglang = None
        self._ray = None
        self._engine = None
        self._tokenizer = None
        self._workers: list[Any] = []

        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )

        if self.dp_size == 1:
            self._init_single()
        else:
            self._init_ray()

    def _import_sglang(self) -> Any:
        try:
            import sglang
        except ImportError as exc:
            raise RuntimeError(
                "sglang is not installed. Install dependencies first (`pip install -e .[sglang]`)."
            ) from exc
        self._sglang = sglang
        return sglang

    def _engine_args(self) -> dict[str, Any]:
        engine_args: dict[str, Any] = {
            "model_path": self.model,
            "tp_size": self.tensor_parallel_size,
        }
        engine_args.update(
            {
                key: value
                for key, value in self.model_kwargs.items()
                if key != "generation_batch_size"
            }
        )
        return engine_args

    def _init_single(self) -> None:
        sglang = self._import_sglang()
        self._engine = sglang.Engine(**self._engine_args())
        self._tokenizer = load_chat_tokenizer(self.model, self.model_kwargs)

    def _init_ray(self) -> None:
        self._import_sglang()
        try:
            import ray
        except ImportError as exc:
            raise RuntimeError(
                "ray is not installed. Install dependencies first (`pip install -e .`)."
            ) from exc

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._ray = ray
        engine_args = self._engine_args()
        model = self.model
        model_kwargs = self.model_kwargs
        generation_batch_size = self.generation_batch_size

        @ray.remote(num_gpus=self.tensor_parallel_size)
        class _Worker:
            def __init__(
                self,
                _engine_args: dict[str, Any],
                _model: str,
                _model_kwargs: dict[str, Any],
                _generation_batch_size: int,
            ) -> None:
                import sglang as _sglang

                from aethereval.backends.prompt import (
                    load_chat_tokenizer as _load_chat_tokenizer,
                )

                self._engine = _sglang.Engine(**_engine_args)
                self._tokenizer = _load_chat_tokenizer(_model, _model_kwargs)
                self._generation_batch_size = _generation_batch_size

            def generate(
                self,
                payloads: list[dict[str, Any]],
                gen_cfg: dict[str, Any],
            ) -> list[dict[str, Any]]:
                return _run_generation(
                    engine=self._engine,
                    tokenizer=self._tokenizer,
                    payloads=payloads,
                    gen_cfg=gen_cfg,
                    batch_size=self._generation_batch_size,
                )

            def close(self) -> None:
                if self._engine is None:
                    return
                shutdown = getattr(self._engine, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                else:
                    close = getattr(self._engine, "close", None)
                    if callable(close):
                        close()
                self._engine = None
                self._tokenizer = None

        self._workers = [
            _Worker.remote(engine_args, model, model_kwargs, generation_batch_size)
            for _ in range(self.dp_size)
        ]

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
            assert self._engine is not None
            output_dicts = _run_generation(
                engine=self._engine,
                tokenizer=self._tokenizer,
                payloads=payloads,
                gen_cfg=gen_cfg,
                batch_size=self.generation_batch_size,
            )
        else:
            assert self._ray is not None
            worker_payloads = _split_payloads(payloads, len(self._workers))
            refs = []
            for worker, worker_items in zip(self._workers, worker_payloads):
                refs.append(worker.generate.remote(worker_items, gen_cfg))
            nested = self._ray.get(refs)
            output_dicts = [item for sublist in nested for item in sublist]
            output_dicts.sort(key=lambda x: x["idx"])

        return _outputs_from_dicts(output_dicts)

    def close(self) -> None:
        if self._engine is not None:
            shutdown = getattr(self._engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
            else:
                close = getattr(self._engine, "close", None)
                if callable(close):
                    close()
        self._engine = None
        self._tokenizer = None
        if self._ray is not None and self._workers:
            self._ray.get([worker.close.remote() for worker in self._workers])
        if self._ray is not None and self._ray.is_initialized():
            self._ray.shutdown()
        self._workers = []
