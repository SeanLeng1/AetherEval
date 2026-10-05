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
    except (TypeError, ValueError, ArithmeticError):
        return None
    return number if 0.0 < abs(number) < 1e-3 else None


def _verify_pair(verify: Any, gold: Any, prediction: Any) -> bool:
    # verify() rounds to 6 decimal places, so 2.88e-19 and 5.76e-19 both become 0.
    # Magnitudes below 1e-3 are compared to 6 significant digits instead.
    small_gold = _small_number(gold)
    if small_gold is not None and getattr(prediction, "is_number", False):
        try:
            return math.isclose(small_gold, float(prediction), rel_tol=1e-6)
        except (TypeError, ValueError, ArithmeticError):
            return False
    return bool(verify(gold, prediction, 6))


@lru_cache(maxsize=1)
def _math_verify_tools() -> tuple[
    Any,
    Any,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
]:
    try:
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
    return parse, verify, latex_target, expr_target, pred_target


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


def format_extractions(values: list[Any]) -> list[str]:
    """Render diagnostics without letting SymPy's printer abort scoring."""
    rendered = []
    for value in values:
        try:
            rendered.append(str(value))
        except (ArithmeticError, RecursionError) as error:
            # Ordering an Add can evaluate its terms (e.g. csc(0)). Keep the
            # expression for verification; this string is only output metadata.
            rendered.append(f"<unprintable {type(value).__name__}: {type(error).__name__}>")
    return rendered


def _matches(verify: Any, golds: list[Any], predictions: list[Any]) -> bool:
    return any(_verify_pair(verify, g, p) for g in golds for p in predictions)


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

    parse() and verify() absorb their own timeouts and errors (returning [] and
    False). They raise only when called off the main thread, which must fail loudly
    instead of scoring every record 0. _verify_pair's small-gold comparison catches
    numeric conversion failures from float(). Formatting extracted expressions
    is diagnostic only and must not change the mathematical comparison.
    """
    # Normalize dataset gold notation during preparation, not arbitrary model text.
    gold_text = str(gold).strip()
    gold_input = f"\\boxed{{{gold_text}}}" if boxed_gold else gold_text
    parse, verify, latex_target, expr_target, pred_target = _math_verify_tools()

    prediction = _normalize_boxed_e_notation(prediction)
    extracted_predictions = parse(prediction, pred_target)
    extracted_golds = parse(gold_input, latex_target)
    if not boxed_gold and not extracted_golds:
        extracted_golds = parse(gold_input, expr_target)

    pred_strings = format_extractions(extracted_predictions)
    gold_strings = format_extractions(extracted_golds)

    if not extracted_golds:
        return 0.0, pred_strings, gold_strings, "no gold extraction"
    if not extracted_predictions:
        return 0.0, pred_strings, gold_strings, None

    matched = _matches(verify, extracted_golds, extracted_predictions)
    if not matched and keep_units_fallback:
        unit_predictions = parse(prediction, _keep_units_pred_target())
        matched = _matches(verify, extracted_golds, unit_predictions)
        if matched:
            pred_strings = format_extractions(unit_predictions)

    return (1.0 if matched else 0.0), pred_strings, gold_strings, None
