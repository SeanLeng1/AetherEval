import math
import re
from dataclasses import replace
from functools import lru_cache
from typing import Any

# math-verify reads \boxed{4.5e33} as 4.5*e*33. Only a box whose entire content is
# e-notation is rewritten; free text is never touched, because an undelimited
# "2.88 \times 10^{-19}" would make the expression extractor pick up the bare "10".
_BOXED_E_NOTATION_RE = re.compile(
    r"\\boxed\{\s*(-?\d+(?:\.\d+)?)[eE]([+-]?\d+)\s*\}"
)


def _normalize_boxed_e_notation(text: str) -> str:
    return _BOXED_E_NOTATION_RE.sub(
        lambda m: f"\\boxed{{{m.group(1)} \\times 10^{{{int(m.group(2))}}}}}", text
    )


def _small_number(value: Any) -> float | None:
    if isinstance(value, str) or not getattr(value, "is_number", False):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 < abs(number) < 1e-3 else None


def _verify_pair(verify: Any, gold: Any, prediction: Any) -> bool:
    # verify() rounds to 6 decimal places, so 2.88e-19 and 5.76e-19 both become 0.
    # Magnitudes below 1e-3 are compared to 6 significant digits instead.
    small_gold = _small_number(gold)
    if small_gold is not None and getattr(prediction, "is_number", False):
        try:
            return math.isclose(small_gold, float(prediction), rel_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return bool(verify(gold, prediction, 6))


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


@lru_cache(maxsize=1)
def _keep_units_pred_target() -> tuple[Any, ...]:
    # math-verify strips trailing unit words, including single letters such as
    # t, m and c, so "\frac{37}{4} m" parses as 37/4. Qwen2.5-Math likewise skips
    # unit removal for Minerva; every other normalization stays at its default.
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    normalization = replace(LatexExtractionConfig().normalization_config, units=False)
    return (
        ExprExtractionConfig(),
        LatexExtractionConfig(normalization_config=normalization),
    )


def score_with_math_verify(
    gold: str,
    prediction: str,
    *,
    boxed_gold: bool = False,
    keep_units_fallback: bool = False,
) -> tuple[float, list[str], list[str], str | None]:
    """Score a generated math answer using math-verify.

    `boxed_gold=True` preserves AIME-style datasets where `gold` is just the final
    answer. Eval-set math tasks pass full `solution` text directly.
    `keep_units_fallback=True` re-parses an unmatched prediction without unit
    stripping, so symbolic answers ending in a variable can still match.
    """
    # Normalize dataset gold notation during preparation, not arbitrary model text.
    gold_text = str(gold).strip()
    gold_input = f"\\boxed{{{gold_text}}}" if boxed_gold else gold_text
    timeout_error, parse, verify, latex_target, expr_target, pred_target = (
        _math_verify_tools()
    )

    prediction = _normalize_boxed_e_notation(prediction)
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
            _verify_pair(verify, g, p)
            for g in extracted_golds
            for p in extracted_predictions
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

    if not matched and keep_units_fallback:
        try:
            unit_predictions = parse(prediction, _keep_units_pred_target())
            matched = any(
                _verify_pair(verify, g, p)
                for g in extracted_golds
                for p in unit_predictions
            )
        except timeout_error:
            return 0.0, pred_strings, gold_strings, "unit fallback timeout"
        except Exception as exc:  # noqa: BLE001
            return (
                0.0,
                pred_strings,
                gold_strings,
                f"unit fallback error: {type(exc).__name__}: {exc}",
            )
        if matched:
            pred_strings = [str(x) for x in unit_predictions]

    return (1.0 if matched else 0.0), pred_strings, gold_strings, None
