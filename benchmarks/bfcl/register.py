"""Register the ToolRL/GDPO model with bfcl_eval's MODEL_CONFIG_MAPPING.

bfcl_eval resolves ``--model <name>`` to a handler via ``MODEL_CONFIG_MAPPING``. We
inject an entry at runtime (in-process, before calling generate/evaluate) so no edit to
the installed package is needed. ``is_fc_model=False`` => prompt mode (our model emits
the ToolRL text format, not native function-calling JSON args).
"""

from ._compat import ensure_bfcl_importable

DEFAULT_REGISTRY_NAME = "rlla-qwen"


def register_rlla_model(
    registry_name: str = DEFAULT_REGISTRY_NAME,
    *,
    is_fc_model: bool = False,
    project_root: str | None = None,
) -> str:
    """Add ``registry_name`` -> RLLAHandler to MODEL_CONFIG_MAPPING (idempotent).

    bfcl loads weights from ``registry_name`` itself (it becomes the handler's
    ``model_name_huggingface``) unless a local checkpoint dir is passed via
    ``--local-model-path``. So use the HF id as ``registry_name`` for HF models, or any
    name together with ``model_path`` for a local checkpoint. (``ModelConfig.model_name``
    is unused by the OSS handler in BFCL v3.)
    """
    ensure_bfcl_importable(project_root=project_root)

    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    from .handler import RLLAHandler

    config = ModelConfig(
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
    MODEL_CONFIG_MAPPING[registry_name] = config

    # BFCL's eval runner reconstructs the handler key from the result-folder name by reversing
    # '/'->'_' (eval_runner.runner: `model_name_escaped = model_name.replace("_", "/")`), assuming
    # every '_' in the folder came from a '/'. A checkpoint whose leaf legitimately contains '_'
    # (e.g. ...-olora_l1_0.5-l2_0.0-parts3_200_huggingface) round-trips to a *different* string with
    # those '_' turned into '/', which was never registered -> KeyError in get_handler at eval time
    # (generation is unaffected). Alias the config under that reconstructed key; it only builds the
    # eval handler, whose model_name is inert (decode_ast/decode_execute never read it).
    runner_key = registry_name.replace("/", "_").replace("_", "/")
    if runner_key != registry_name:
        MODEL_CONFIG_MAPPING[runner_key] = config
    return registry_name
