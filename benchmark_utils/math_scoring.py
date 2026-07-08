from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _math_verify_tools() -> tuple[
    Any,
    Any,
    Any,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
]:
    try:
        from math_verify.errors import TimeoutException
        from math_verify.grader import verify
        from math_verify.parser import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "math-verify is required for math metrics. Install with `pip install math-verify`."
        ) from exc

    latex_target = (LatexExtractionConfig(),)
    expr_target = (ExprExtractionConfig(),)
    pred_target = (ExprExtractionConfig(), LatexExtractionConfig())
    return TimeoutException, parse, verify, latex_target, expr_target, pred_target


def score_with_math_verify(
    gold: str,
    prediction: str,
    *,
    boxed_gold: bool = False,
) -> tuple[float, list[str], list[str], str | None]:
    """Score a generated math answer using math-verify.

    `boxed_gold=True` preserves AIME-style datasets where `gold` is just the final
    answer. Eval-set math tasks pass full `solution` text directly.
    """
    gold_text = str(gold).strip()
    gold_input = f"\\boxed{{{gold_text}}}" if boxed_gold else gold_text
    timeout_error, parse, verify, latex_target, expr_target, pred_target = (
        _math_verify_tools()
    )

    try:
        extracted_predictions = parse(prediction, pred_target)
        extracted_golds = parse(gold_input, latex_target)
        if not boxed_gold and not extracted_golds:
            extracted_golds = parse(gold_input, expr_target)
    except timeout_error:
        return 0.0, [], [], "parse timeout"
    except Exception as exc:  # noqa: BLE001
        return 0.0, [], [], f"parse error: {type(exc).__name__}: {exc}"

    pred_strings = [str(x) for x in extracted_predictions]
    gold_strings = [str(x) for x in extracted_golds]

    if not extracted_golds:
        return 0.0, pred_strings, gold_strings, "no gold extraction"
    if not extracted_predictions:
        return 0.0, pred_strings, gold_strings, None

    try:
        matched = any(
            verify(g, p, 6) for g in extracted_golds for p in extracted_predictions
        )
    except timeout_error:
        return 0.0, pred_strings, gold_strings, "verify timeout"
    except Exception as exc:  # noqa: BLE001
        return (
            0.0,
            pred_strings,
            gold_strings,
            f"verify error: {type(exc).__name__}: {exc}",
        )

    return (1.0 if matched else 0.0), pred_strings, gold_strings, None
