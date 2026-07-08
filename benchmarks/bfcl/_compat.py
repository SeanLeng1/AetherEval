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
import os
import sys
import tempfile
import types
from pathlib import Path


_TREE_SITTER_MODULES = (
    "tree_sitter",
    "tree_sitter_java",
    "tree_sitter_javascript",
)
_TREE_SITTER_DEPENDENTS = (
    "bfcl_eval.model_handler.parser.java_parser",
    "bfcl_eval.model_handler.parser.js_parser",
    "bfcl_eval.model_handler.utils",
    "bfcl_eval.constants.model_config",
)
_TENACITY_DEPENDENTS = (
    "bfcl_eval.model_handler.utils",
    "bfcl_eval.model_handler.api_inference.claude",
    "bfcl_eval.model_handler.api_inference.cohere",
    "bfcl_eval.constants.model_config",
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


class _TreeSitterLanguage:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _TreeSitterNode:
    type = ""
    children: list = []
    child_count = 0
    start_byte = 0
    end_byte = 0
    text = b""

    def child_by_field_name(self, name: str):
        return None

    def sexp(self) -> str:
        return ""


class _TreeSitterTree:
    root_node = _TreeSitterNode()


class _TreeSitterParser:
    def __init__(self) -> None:
        self.language = None

    def set_language(self, language) -> None:
        self.language = language

    def parse(self, source) -> _TreeSitterTree:
        return _TreeSitterTree()


def _ensure_tree_sitter_stubs() -> None:
    tree_sitter = types.ModuleType("tree_sitter")
    tree_sitter.Language = _TreeSitterLanguage
    tree_sitter.Parser = _TreeSitterParser
    sys.modules["tree_sitter"] = tree_sitter

    for module_name in ("tree_sitter_java", "tree_sitter_javascript"):
        module = types.ModuleType(module_name)
        module.language = lambda: object()
        sys.modules[module_name] = module

    for module_name in _TREE_SITTER_DEPENDENTS:
        sys.modules.pop(module_name, None)


class _RetryCondition:
    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self


def _retry(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


def _retry_condition(*args, **kwargs) -> _RetryCondition:
    return _RetryCondition()


def _tenacity_value(*args, **kwargs):
    return object()


def _ensure_tenacity_stubs() -> None:
    tenacity = types.ModuleType("tenacity")
    tenacity.retry = _retry
    tenacity.retry_if_exception_message = _retry_condition
    tenacity.retry_if_exception_type = _retry_condition
    tenacity.wait_random_exponential = _tenacity_value
    sys.modules["tenacity"] = tenacity

    stop = types.ModuleType("tenacity.stop")
    stop.stop_after_attempt = _tenacity_value
    sys.modules["tenacity.stop"] = stop

    for module_name in _TENACITY_DEPENDENTS:
        sys.modules.pop(module_name, None)


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
    """Import bfcl_eval.constants.model_config, patching SDK gaps until it succeeds."""
    target = "bfcl_eval.constants.model_config"
    _set_bfcl_project_root(project_root)
    _ensure_tree_sitter_stubs()
    _ensure_tenacity_stubs()
    for _ in range(max_iters):
        try:
            importlib.import_module(target)
            return
        except ModuleNotFoundError as exc:
            missing = exc.name
            if not missing or missing.startswith("bfcl_eval"):
                raise
            if missing in _TREE_SITTER_MODULES:
                _ensure_tree_sitter_stubs()
            elif missing == "tenacity" or missing.startswith("tenacity."):
                _ensure_tenacity_stubs()
            else:
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
