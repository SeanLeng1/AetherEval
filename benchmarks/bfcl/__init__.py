"""BFCL-v3 external benchmark extension for AetherEval (wraps bfcl_eval)."""

from .register import DEFAULT_REGISTRY_NAME, register_rlla_model

__all__ = ["DEFAULT_REGISTRY_NAME", "register_rlla_model"]
