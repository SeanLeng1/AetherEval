"""Adapt SGLang's legacy OLMo3 config to Transformers 5 checkpoint metadata."""

import inspect
from functools import wraps


def install_olmo3_config_compatibility() -> None:
    from sglang.srt.configs.olmo3 import Olmo3Config

    original = Olmo3Config.__init__
    if getattr(original, "_aethereval_rope_compat", False):
        return
    parameters = inspect.signature(original).parameters
    # Only the legacy constructor consumes rope_scaling after calling super().
    if "rope_scaling" not in parameters or "rope_parameters" in parameters:
        return

    @wraps(original)
    def compatible_init(self, *args, **kwargs):
        rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_parameters is not None:
            legacy = kwargs.get("rope_scaling")
            if legacy is not None and legacy != rope_parameters:
                raise ValueError("Conflicting OLMo3 rope_scaling and rope_parameters")
            kwargs["rope_scaling"] = dict(rope_parameters)
            if "rope_theta" in rope_parameters:
                kwargs["rope_theta"] = rope_parameters["rope_theta"]
        # Defer TF5's RoPE validation until max_position_embeddings is initialized.
        original(self, *args, **kwargs)
        if rope_parameters is not None:
            # SGLang's OLMo2 implementation reads rope_scaling, while Transformers
            # reads rope_parameters. Preserve the full YaRN settings in both.
            self.rope_parameters = dict(rope_parameters)
            validate = getattr(self, "validate_rope", None)
            if validate is not None:
                validate()

    compatible_init._aethereval_rope_compat = True
    Olmo3Config.__init__ = compatible_init
