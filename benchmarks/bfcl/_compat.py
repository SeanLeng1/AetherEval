"""Make bfcl_eval importable under drifted optional-API-SDK versions.

``bfcl_eval.constants.model_config`` eagerly imports every handler, including the
API-inference ones (cohere / anthropic / together / ...). Those SDKs are pinned by
bfcl-eval but we install it ``--no-deps`` to avoid disturbing the container's sglang
stack, so their newer/missing versions raise at import. We only ever use the local
sglang OSS handler, so we stub whatever the API handlers need: an import-retry loop
fills missing modules with a permissive fake and missing attributes with dummy types,
until ``model_config`` imports. No real API SDK behaviour is required.
"""

import importlib
import sys
import types


class _LaxModule(types.ModuleType):
    """Module whose missing attributes resolve to fresh dummy types (cached)."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dummy = type(name, (), {})
        setattr(self, name, dummy)
        return dummy


def _ensure_lax_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if isinstance(mod, _LaxModule):
        return mod
    lax = _LaxModule(name)
    sys.modules[name] = lax
    return lax


def ensure_bfcl_importable(max_iters: int = 64) -> None:
    """Import bfcl_eval.constants.model_config, patching SDK gaps until it succeeds."""
    target = "bfcl_eval.constants.model_config"
    for _ in range(max_iters):
        try:
            importlib.import_module(target)
            return
        except ModuleNotFoundError as exc:
            missing = exc.name
            if not missing or missing.startswith("bfcl_eval"):
                raise
            _ensure_lax_module(missing)
        except (AttributeError, ImportError) as exc:
            # e.g. "module 'cohere.types' has no attribute 'ChatResponse'"
            msg = str(exc)
            if "has no attribute" in msg and "'" in msg:
                parts = msg.split("'")
                mod_name, attr = parts[1], parts[3]
                mod = sys.modules.get(mod_name)
                if mod is None:
                    mod = _ensure_lax_module(mod_name)
                setattr(mod, attr, type(attr, (), {}))
            else:
                raise
    raise ImportError(f"could not make {target} importable after {max_iters} patches")
