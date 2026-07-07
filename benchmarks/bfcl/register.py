"""Register the ToolRL/GDPO model with bfcl_eval's MODEL_CONFIG_MAPPING.

bfcl_eval resolves ``--model <name>`` to a handler via ``MODEL_CONFIG_MAPPING``. We
inject an entry at runtime (in-process, before calling generate/evaluate) so no edit to
the installed package is needed. ``is_fc_model=False`` => prompt mode (our model emits
the ToolRL text format, not native function-calling JSON args).
"""

from ._compat import ensure_bfcl_importable

ensure_bfcl_importable()  # stub drifted optional API SDKs before model_config imports

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

from .handler import RLLAHandler

DEFAULT_REGISTRY_NAME = "rlla-qwen"


def register_rlla_model(
    registry_name: str = DEFAULT_REGISTRY_NAME,
    *,
    is_fc_model: bool = False,
) -> str:
    """Add ``registry_name`` -> RLLAHandler to MODEL_CONFIG_MAPPING (idempotent).

    bfcl loads weights from ``registry_name`` itself (it becomes the handler's
    ``model_name_huggingface``) unless a local checkpoint dir is passed via
    ``--local-model-path``. So use the HF id as ``registry_name`` for HF models, or any
    name together with ``model_path`` for a local checkpoint. (``ModelConfig.model_name``
    is unused by the OSS handler in BFCL v3.)
    """
    MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
        model_name=registry_name,
        display_name=f"{registry_name} (ToolRL/GDPO)",
        url="https://github.com/SeanLeng1/AetherEval",
        org="AetherRL",
        license="Apache 2.0",
        model_handler=RLLAHandler,
        input_price=None,
        output_price=None,
        is_fc_model=is_fc_model,
        underscore_to_dot=False,
    )
    return registry_name
