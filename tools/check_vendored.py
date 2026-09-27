"""Show the code-level diff between a vendored file and its pinned upstream source.

Both sides are parsed, stripped of docstrings and re-rendered with ast.unparse, so
formatting, comments and docstrings drop out and only real code changes remain
(for example rewritten imports or removed definitions). Commands for each vendored
file are listed in its folder's NOTICE.

Usage: python tools/check_vendored.py <upstream path or raw URL> <local path>
Exit status is 1 when any code difference remains.
"""

import ast
import difflib
import sys
import urllib.request
from pathlib import Path


def _read(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def normalize(text: str) -> list[str]:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).splitlines()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    upstream, local = argv
    diff = list(
        difflib.unified_diff(
            normalize(_read(upstream)),
            normalize(_read(local)),
            fromfile=upstream,
            tofile=local,
            lineterm="",
        )
    )
    print("\n".join(diff))
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
