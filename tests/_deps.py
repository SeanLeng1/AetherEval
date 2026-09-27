"""Skip tests whose optional scorer or runtime dependencies are not installed.

Set AETHEREVAL_REQUIRE_ALL_TEST_DEPS=1 (e.g. in the runtime image) to run them
anyway, so a missing dependency fails instead of skipping.
"""

import importlib.util
import os
import unittest

# bfcl_eval plus the modules benchmarks.bfcl._compat.ensure_bfcl_importable requires.
BFCL_MODULES = (
    "bfcl_eval",
    "tree_sitter",
    "tree_sitter_java",
    "tree_sitter_javascript",
    "tenacity",
)


def _missing(modules: tuple[str, ...]) -> list[str]:
    if os.environ.get("AETHEREVAL_REQUIRE_ALL_TEST_DEPS") == "1":
        return []
    return [name for name in modules if importlib.util.find_spec(name) is None]


def available(*modules: str) -> bool:
    return not _missing(modules)


def requires(*modules: str):
    missing = _missing(modules)
    return unittest.skipIf(
        bool(missing), f"optional dependency missing: {', '.join(missing)}"
    )
