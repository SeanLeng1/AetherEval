"""EvalPlus's official sanitize with an exact, pruned ``code_extract``.

Upstream ``code_extract`` (pinned 26d6d00) parses every line span (i, j), which is
cubic in response length: a truncated 16k-token code answer can stall scoring for
minutes. This version returns the identical string, namely the first span in
upstream's (i, j) order with the most non-blank lines among parseable spans of at
least two lines; tests/test_evalplus_sanitize.py checks it against upstream.
It skips spans too short to win, which removes the cubic scan when the code runs
to near the end (assert floods, unclosed fences); code followed by long prose is
as slow as upstream.
"""

import re

import evalplus.sanitize as _upstream
from evalplus.syncheck import syntax_check

UPSTREAM_CODE_EXTRACT = _upstream.code_extract
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def code_extract(text: str) -> str:
    lines = _ANSI_ESCAPE.sub("", text).split("\n")
    nonblank = [0]  # nonblank[k] counts the non-blank lines in lines[:k]
    for line in lines:
        nonblank.append(nonblank[-1] + bool(line.strip()))

    def count(i: int, j: int) -> int:
        return nonblank[j + 1] - nonblank[i]

    best, pair = 0, (0, 0)
    for i in range(len(lines)):
        # Counts only shrink as i grows, and upstream keeps only strictly longer spans.
        if count(i, len(lines) - 1) <= best:
            break
        # The largest parseable j gives this start's maximum count.
        top = None
        for j in range(len(lines) - 1, i, -1):
            if count(i, j) <= best:
                break
            if syntax_check("\n".join(lines[i : j + 1])):
                top = j
                break
        if top is None:
            continue
        # Upstream keeps the smallest parseable j with that count (trailing blanks differ).
        low = top
        while low - 1 > i and count(i, low - 1) == count(i, top):
            low -= 1
        end = next(k for k in range(low, top + 1)
                   if k == top or syntax_check("\n".join(lines[i : k + 1])))
        best, pair = count(i, top), (i, end)
    return "\n".join(lines[pair[0] : pair[1] + 1])


# sanitize() looks code_extract up as a module global.
_upstream.code_extract = code_extract
def sanitize(code: str, entrypoint: str | None = None) -> str:
    """Upstream sanitize, or "" when the code nests too deeply for its recursive
    dependency walk (e.g. a thousand-term ``a + b + ...``); an empty solution then
    fails every test instead of aborting the evaluation."""
    try:
        return _upstream.sanitize(code, entrypoint=entrypoint)
    except RecursionError:
        return ""
