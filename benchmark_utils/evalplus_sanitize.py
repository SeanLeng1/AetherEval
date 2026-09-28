"""EvalPlus's official sanitize with an exact, bounded ``code_extract``.

Upstream ``code_extract`` (pinned 26d6d00) parses every line span (i, j), which is
cubic in response length: a degenerate 16k-token response can stall scoring for
hours. This version returns the identical string, namely the first span in
upstream's (i, j) order with the most non-blank lines among parseable spans of at
least two lines; tests/test_evalplus_sanitize.py checks it against upstream.

Two prunes remove the cubic blowups without changing the result: spans that
cannot beat the best count are skipped, and one parse of each start's whole tail
bounds where its spans can end (spans running past a syntax error share the
failing prefix, and an opener the tail never closes has no closer in any span).
A deterministic parse budget backstops adversarial inputs; exhausting it keeps
the best span found so far and prints a warning.
"""

import ast
import re

import evalplus.sanitize as _upstream
from evalplus.syncheck import syntax_check

UPSTREAM_CODE_EXTRACT = _upstream.code_extract
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
# Characters handed to the parser before code_extract stops refining its answer.
# Ordinary responses stay far below a million; only adversarial text comes close.
_PARSE_BUDGET_CHARS = 100_000_000


class _ParseBudgetExhausted(Exception):
    pass


def _insignificant(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def code_extract(text: str) -> str:
    lines = _ANSI_ESCAPE.sub("", text).split("\n")
    cleaned = "\n".join(lines)
    n = len(lines)
    offsets = [0]
    nonblank = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
        nonblank.append(nonblank[-1] + bool(line.strip()))

    def span(i: int, j: int) -> str:
        return cleaned[offsets[i] : offsets[j + 1] - 1]

    def count(i: int, j: int) -> int:
        return nonblank[j + 1] - nonblank[i]

    budget = _PARSE_BUDGET_CHARS

    def charge(chars: int) -> None:
        nonlocal budget
        budget -= chars
        if budget < 0:
            raise _ParseBudgetExhausted

    def check(i: int, j: int) -> bool:
        candidate = span(i, j)
        charge(len(candidate))
        return syntax_check(candidate)

    def probe(i: int) -> tuple[bool, int]:
        """Parse the tail once: (tail parses, largest j a span from i can reach).

        Spans running two or more lines past the tail's syntax error share the
        failing prefix, and an opener "never closed" in the tail has no closer in
        any span either. Blank and comment lines carry no tokens, so the bound
        extends across them and one significant line beyond the error.
        """
        try:
            ast.parse(cleaned[offsets[i] :])
        except SyntaxError as exc:
            end = exc.end_lineno or exc.lineno
            if end is None:
                charge(offsets[n] - offsets[i])
                return False, n - 1
            cap = min(n - 1, i + end + 1)
            while cap < n - 1 and _insignificant(lines[cap + 1]):
                cap += 1
            cap = min(n - 1, cap + 1)
            charge(offsets[cap + 1] - offsets[i])
            return False, cap
        except MemoryError:
            charge(offsets[n] - offsets[i])
            return False, n - 1
        charge(offsets[n] - offsets[i])
        return True, n - 1

    best, pair = 0, (0, 0)
    try:
        for i in range(n):
            # Counts only shrink as i grows, and upstream keeps strictly longer spans.
            if count(i, n - 1) <= best:
                break
            parses, cap = probe(i)
            top = None
            if parses:
                if n - 1 > i:
                    top = n - 1
            else:
                # The largest parseable j gives this start's maximum count.
                for j in range(cap, i, -1):
                    if count(i, j) <= best:
                        break
                    if check(i, j):
                        top = j
                        break
            if top is None:
                continue
            # Upstream keeps the smallest parseable j with that count (trailing blanks differ).
            low = top
            while low - 1 > i and count(i, low - 1) == count(i, top):
                low -= 1
            end = next(
                k for k in range(low, top + 1) if k == top or check(i, k)
            )
            best, pair = count(i, top), (i, end)
    except _ParseBudgetExhausted:
        print(
            "[aethereval] code_extract stopped at its "
            f"{_PARSE_BUDGET_CHARS}-character parse budget; "
            "keeping the best span found so far"
        )
    return span(pair[0], pair[1])


# sanitize() looks code_extract up as a module global.
_upstream.code_extract = code_extract


def sanitize(code: str, entrypoint: str | None = None) -> str:
    """Upstream sanitize, or "" when its machinery rejects the code itself: too
    deeply nested for the recursive dependency walk (a thousand-term ``a + b +
    ...``), or a null byte on Python < 3.12, where ast raises ValueError. An
    empty solution then fails every test instead of aborting the evaluation."""
    try:
        return _upstream.sanitize(code, entrypoint=entrypoint)
    except (RecursionError, ValueError):
        return ""
