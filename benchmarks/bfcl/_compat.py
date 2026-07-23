"""Make BFCL's eager model registry tolerate unused provider SDKs.

``bfcl_eval.constants.model_config`` eagerly imports every handler, including the
API-inference ones. AetherEval only registers its local SGLang handler, so missing
provider-only modules may be represented by permissive placeholders. Core BFCL v4
parser dependencies are never stubbed: doing so would silently corrupt Java and
JavaScript scores.
"""

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path


_REQUIRED_MODULES = (
    "tree_sitter",
    "tree_sitter_java",
    "tree_sitter_javascript",
    "tenacity",
)


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


def _set_bfcl_project_root(project_root: str | Path | None) -> None:
    if project_root is not None:
        os.environ["BFCL_PROJECT_ROOT"] = str(Path(project_root).resolve())
        return
    os.environ.setdefault(
        "BFCL_PROJECT_ROOT",
        str((Path(tempfile.gettempdir()) / "aethereval_bfcl").resolve()),
    )


def ensure_bfcl_importable(
    max_iters: int = 64,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Import BFCL's registry while stubbing only unused provider SDKs."""
    target = "bfcl_eval.constants.model_config"
    _set_bfcl_project_root(project_root)
    for module_name in _REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "BFCL v4 runtime dependency is missing: "
                    f"{module_name}. Rebuild the SGLang runtime image with the "
                    "BFCL v4 dependencies."
                ) from exc
    for _ in range(max_iters):
        try:
            importlib.import_module(target)
            return
        except ModuleNotFoundError as exc:
            missing = exc.name
            if not missing or missing.startswith("bfcl_eval"):
                raise
            if missing in _REQUIRED_MODULES:
                raise RuntimeError(
                    f"BFCL v4 runtime dependency is missing: {missing}."
                ) from exc
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
            elif msg.startswith("No ") and " module name -> " in msg:
                attr = msg.split("No ", 1)[1].split(" found ", 1)[0]
                mod_name = msg.rsplit(" -> ", 1)[1]
                mod = sys.modules.get(mod_name)
                if mod is None:
                    mod = _ensure_lax_module(mod_name)
                setattr(mod, attr, type(attr, (), {}))
            elif msg.startswith("cannot import name") and "'" in msg:
                parts = msg.split("'")
                attr = parts[1]
                mod_name = parts[3]
                mod = sys.modules.get(mod_name)
                if mod is None:
                    mod = _ensure_lax_module(mod_name)
                setattr(mod, attr, type(attr, (), {}))
            else:
                raise
    raise ImportError(f"could not make {target} importable after {max_iters} patches")
