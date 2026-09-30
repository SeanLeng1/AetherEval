"""Symbolic grading for the five OCW answers unsupported by math-verify."""

import re
from dataclasses import replace

from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from math_verify.utils import timeout
from sympy import Basic, Derivative, Equality, Poly, nan, oo, zoo
from sympy.polys.polyerrors import PolynomialError

REPAIRED_SAMPLES = frozenset(
    f"minervamath_{index}" for index in (27, 138, 261, 268, 269)
)


def _boxed_answer(text: str) -> str:
    """Take the last complete box, preserving its nested LaTeX braces."""
    answers = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] in "{}" and text[index - 1] != "\\":
                depth += 1 if text[index] == "{" else -1
            if depth == 0:
                answers.append(text[start:index])
                break
    if not answers:
        raise ValueError("Minerva symbolic reference has no complete boxed answer")
    return answers[-1]


def _symbol_aliases(reference: str, prediction: str) -> tuple[str, str]:
    # These are variables: E_n is not Euler's constant, gamma is not the Gamma
    # function, and the solar subscript is an identifier. Rename both sides with
    # a fresh prefix so a candidate's existing variable cannot collide with it.
    source = re.sub(r"\s+", "", reference + prediction).lower()
    prefix = "aevalphysics"
    while prefix in source:
        prefix += "x"

    def normalize(text: str) -> str:
        if re.search(r"(?<![A-Za-z\\])e(?![A-Za-z])", reference):
            # In the oscillator answer e denotes electric charge, not exp(1).
            text = re.sub(
                r"(?<![A-Za-z\\])e(?=E\s*_|[^A-Za-z]|$)",
                lambda _: rf"\text{{{prefix}charge}}",
                text,
            )
        text = re.sub(
            r"(?<![A-Za-z\\])E\s*(?=_)",
            lambda _: rf"\text{{{prefix}energy}}",
            text,
        )
        text = re.sub(
            r"\\gamma\b|γ", lambda _: rf"\text{{{prefix}gamma}}", text
        )
        return re.sub(
            r"([LM])\s*_\s*(?:\{\s*\\odot\s*\}|\\odot\b)",
            lambda match: rf"\text{{{prefix}solar{match[1].lower()}}}",
            text,
        )

    return normalize(reference), normalize(prediction)


def _expressions(text: str, extraction: LatexExtractionConfig) -> list[Basic]:
    return [
        value
        for value in parse(text, (extraction,))
        if isinstance(value, Basic) and not value.has(nan, zoo, oo, -oo)
    ]


@timeout(5)
def _same_answer(reference: Basic, prediction: Basic) -> bool:
    if isinstance(reference, Equality) and isinstance(reference.lhs, Derivative):
        # The mass-growth answer may be the rate alone or an equivalent linear
        # ODE. Keep the derivative as an independent generator; evaluating
        # d(Symbol("M"))/dt would incorrectly collapse it to zero.
        rate = reference.lhs
        if isinstance(prediction, Equality):
            try:
                polynomial = Poly(prediction.lhs - prediction.rhs, rate)
            except PolynomialError:
                return False
            if polynomial.degree() != 1:
                return False
            prediction = -polynomial.nth(0) / polynomial.nth(1)
        reference = reference.rhs
    return bool(verify(reference, prediction, 6))


def score_symbolic_answer(gold: str, prediction: str) -> dict:
    reference, prediction = _symbol_aliases(_boxed_answer(gold), prediction)
    base = LatexExtractionConfig()
    extraction = replace(
        base,
        boxed_match_priority=0,
        normalization_config=replace(base.normalization_config, units=False),
    )
    references = _expressions(r"\boxed{" + reference + "}", extraction)
    if not references:
        raise ValueError("Minerva symbolic reference did not parse to a finite expression")
    predictions = _expressions(prediction, extraction)
    matched = False
    for expected in references:
        for actual in predictions:
            try:
                matched = _same_answer(expected, actual)
            except TimeoutException:
                matched = False
            if matched:
                break
        if matched:
            break
    prediction_strings = [str(value) for value in predictions]
    return {
        "score": float(matched),
        "is_pass": matched,
        "parsed": {
            "prediction_extracted": prediction_strings,
            "gold_extracted": [str(value) for value in references],
        },
        "meta": {
            "prediction_extracted": prediction_strings[0] if predictions else None,
            "symbolic_physics": True,
        },
    }
