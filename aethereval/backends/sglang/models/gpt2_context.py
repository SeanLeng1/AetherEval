"""Inclusive input limits for GPT-2 classification; generation remains upstream."""

import functools
import inspect
import sys
import textwrap
from types import FunctionType

_MARKER = "__aether_gpt2_context__"


def _is_gpt2(config):
    return config.hf_config.architectures == ["GPT2ForSequenceClassification"] and not config.is_generation


def _tokenizer_validation(original):
    source = textwrap.dedent(inspect.getsource(original))
    anchor = "if input_token_num >= self.context_len:"
    if source.count(anchor) != 1:
        raise RuntimeError("SGLang tokenizer length check changed; review the GPT-2 classification patch")
    namespace = dict(original.__globals__)
    source = source.replace(anchor, "if input_token_num > self.context_len:")
    exec(compile(source, inspect.getfile(original), "exec"), namespace)
    inclusive = namespace[original.__name__]

    @functools.wraps(original)
    def validate(self, obj, input_ids):
        from sglang.srt.managers.io_struct import EmbeddingReqInput

        if _is_gpt2(self.model_config) and isinstance(obj, EmbeddingReqInput):
            count = len(input_ids) + self.num_reserved_tokens
            if count > self.context_len:
                raise ValueError(f"GPT-2 classification input has {count} tokens; limit is {self.context_len}")
            return inclusive(self, obj, input_ids)
        return original(self, obj, input_ids)

    return validate


def _worker_info(original):
    @functools.wraps(original)
    def get_info(self):
        info = original(self)
        if not _is_gpt2(self.model_config):
            return info
        runner = self.model_runner
        limit = min(
            self.model_config.context_len,
            runner.req_to_token_pool.max_context_len,
            runner.effective_max_total_num_tokens * self.ps.attn_dcp_size - 1,
        )
        return (*info[:4], limit, limit, *info[6:])

    return get_info


def _validate_classification_length(req, limit, allow_auto_truncate):
    if len(req.origin_input_ids) > limit:
        message = f"GPT-2 classification input has {len(req.origin_input_ids)} tokens; limit is {limit}"
        req.set_finish_with_abort(message)
        return message
    return None


def _embedding_handler(original):
    namespace = {**original.__globals__, "validate_input_length": _validate_classification_length}
    inclusive = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    inclusive.__kwdefaults__ = original.__kwdefaults__

    @functools.wraps(original)
    def handle(self, *args, **kwargs):
        function = inclusive if _is_gpt2(self.model_config) else original
        return function(self, *args, **kwargs)

    return handle


def _patch_loaded_components():
    targets = (
        ("tokenizer_manager", "TokenizerManager", "_validate_one_request", _tokenizer_validation),
        ("tp_worker", "TpModelWorker", "get_worker_info", _worker_info),
        ("scheduler", "Scheduler", "handle_embedding_request", _embedding_handler),
    )
    for module_name, class_name, method_name, factory in targets:
        module = sys.modules.get(f"sglang.srt.managers.{module_name}")
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        original = getattr(cls, method_name)
        if not getattr(original, _MARKER, False):
            replacement = factory(original)
            setattr(replacement, _MARKER, True)
            setattr(cls, method_name, replacement)


def install_context_patch():
    from sglang.srt.configs.model_config import ModelConfig

    if getattr(ModelConfig.__init__, _MARKER, False):
        return
    original = ModelConfig.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if _is_gpt2(self):
            _patch_loaded_components()

    setattr(initialize, _MARKER, True)
    ModelConfig.__init__ = initialize
    # Spawned workers import the model package while manager modules are still loading.
    _patch_loaded_components()
