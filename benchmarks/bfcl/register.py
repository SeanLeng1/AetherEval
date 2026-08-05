"""Select and register a BFCL V3 model-handler profile."""

import copy

from ._compat import ensure_bfcl_importable


HANDLER_PROFILES = ("toolrl", "official")


def _validate_bfcl_v3() -> None:
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX

    if VERSION_PREFIX != "BFCL_v3":
        raise RuntimeError(
            "AetherEval's BFCL adapter requires bfcl-eval==2025.6.8 (BFCL V3) "
            f"(installed dataset prefix: {VERSION_PREFIX!r})."
        )


def _toolrl_config(model_name: str):
    from bfcl_eval.constants.model_config import ModelConfig

    from .handlers import ToolRLHandler

    return ModelConfig(
        model_name=model_name,
        display_name=f"{model_name} (ToolRL)",
        url="https://github.com/SeanLeng1/AetherEval",
        org="AetherRL",
        license="Apache 2.0",
        model_handler=ToolRLHandler,
        input_price=None,
        output_price=None,
        is_fc_model=False,
        underscore_to_dot=False,
    )


def _official_config(model_name: str, model_configs):
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    from .handlers import OfficialPromptHandlerAdapter

    config = model_configs.get(model_name)
    if config is None:
        raise ValueError(
            "BFCL handler profile 'official' requires --model to be an exact "
            "prompt-mode model ID registered by bfcl-eval==2025.6.8; "
            f"{model_name!r} is not registered. Use --bfcl-handler toolrl for "
            "ToolRL-trained or local checkpoints."
        )
    if config.is_fc_model:
        raise ValueError(
            "BFCL handler profile 'official' currently supports upstream "
            "prompt-mode handlers only; native function-calling model "
            f"{model_name!r} is registered as is_fc_model=True."
        )

    upstream_handler = config.model_handler
    if getattr(upstream_handler, "_aethereval_official_adapter", False):
        return config
    if not issubclass(upstream_handler, OSSHandler):
        raise ValueError(
            "BFCL handler profile 'official' supports local prompt-mode OSS "
            f"handlers; {model_name!r} uses {upstream_handler.__name__}."
        )

    adapted_handler = type(
        f"AetherEval{upstream_handler.__name__}",
        (OfficialPromptHandlerAdapter, upstream_handler),
        {
            "__module__": __name__,
            "_aethereval_official_adapter": True,
        },
    )
    adapted = copy.copy(config)
    adapted.model_handler = adapted_handler
    return adapted


def prepare_bfcl_model(
    model_name: str,
    *,
    handler_profile: str = "toolrl",
    project_root: str | None = None,
) -> str:
    """Register ``model_name`` with the selected output-protocol handler.

    ``toolrl`` accepts any Hugging Face ID or local checkpoint path and installs
    AetherEval's ToolRL tag protocol. ``official`` preserves the exact prompt and
    decoder from an existing BFCL V3 prompt-mode model registration.
    """
    if handler_profile not in HANDLER_PROFILES:
        raise ValueError(
            f"Unknown BFCL handler profile {handler_profile!r}; "
            f"choose one of {', '.join(HANDLER_PROFILES)}."
        )

    ensure_bfcl_importable(project_root=project_root)
    _validate_bfcl_v3()

    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    if handler_profile == "toolrl":
        config = _toolrl_config(model_name)
        MODEL_CONFIG_MAPPING[model_name] = config

        # BFCL V3 reconstructs the registry key from its result folder by turning
        # every underscore back into a slash. Local checkpoints can contain both,
        # so register the reconstructed spelling as an evaluator-only alias.
        runner_key = model_name.replace("/", "_").replace("_", "/")
        if runner_key != model_name:
            MODEL_CONFIG_MAPPING[runner_key] = config
    else:
        MODEL_CONFIG_MAPPING[model_name] = _official_config(
            model_name,
            MODEL_CONFIG_MAPPING,
        )

    return model_name
