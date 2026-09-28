import itertools
import random
import unittest

from tests._deps import requires


@requires("evalplus", "tree_sitter_python")
class EvalPlusCodeExtractTests(unittest.TestCase):
    def test_pruned_code_extract_matches_upstream(self) -> None:
        # Imported per test: EvalPlus and tree-sitter are optional scoring installs.
        import evalplus.sanitize as upstream_module

        from benchmark_utils import evalplus_sanitize

        # Also catches semantic drift when the EvalPlus pin is bumped.
        upstream = evalplus_sanitize.UPSTREAM_CODE_EXTRACT
        lines = ["x=1", "", "  ", "def f():", "    return 1", "(", ")", "```",
                 "\x1b[31m", "else:", "# c", "\\", '"""']
        texts = ["\n".join(c) for n in range(5) for c in itertools.product(lines, repeat=n)]
        rnd = random.Random(0)
        texts += ["\n".join(rnd.choices(lines, k=rnd.randint(5, 40))) for _ in range(300)]
        for text in texts:
            self.assertEqual(evalplus_sanitize.code_extract(text), upstream(text), repr(text))
        self.assertIs(upstream_module.code_extract, evalplus_sanitize.code_extract)
        self.assertIsNot(upstream, evalplus_sanitize.code_extract)

    def test_sanitize_matches_upstream_on_markdown_responses(self) -> None:
        import evalplus.sanitize as upstream_module

        from benchmark_utils import evalplus_sanitize

        upstream = evalplus_sanitize.UPSTREAM_CODE_EXTRACT
        code = "import math\n\ndef helper(x):\n    return math.sqrt(x)\n\ndef f(x):\n    return helper(x)\n"
        responses = [
            f"Reasoning first.\n```python\n{code}```\nDone.",
            f"```python\n{code}",  # truncated at the token limit
            f"Plan:\n{code}\nprint(f(4))\nThat is all.",
            "No code here, only prose.\nSecond line.",
            f"```python\n{code}```\n" + "assert f(4) == 2\n" * 30,
        ]
        for text in responses:
            with self.subTest(text=text[:30]):
                self.assertEqual(evalplus_sanitize.code_extract(text), upstream(text))
                for entrypoint in ("f", None):
                    upstream_module.code_extract = upstream
                    try:
                        expected = upstream_module.sanitize(text, entrypoint=entrypoint)
                    finally:
                        upstream_module.code_extract = evalplus_sanitize.code_extract
                    self.assertEqual(
                        evalplus_sanitize.sanitize(text, entrypoint=entrypoint), expected
                    )


    def test_tail_probe_matches_upstream_on_longer_mixed_texts(self) -> None:
        from benchmark_utils import evalplus_sanitize

        upstream = evalplus_sanitize.UPSTREAM_CODE_EXTRACT
        pool = ["x=1", "", "    ", "def f():", "    return 1", "(", ")", "```python",
                "The answer is:", "- first item", "1. step one", "if True:", "else:",
                "import os", "assert f(1) == 2", '"""', "'''", "@wraps", "x)", "# note",
                "    # indented comment", "print(", "return 1", "\\", "y = [", "]",
                'f"{x', 'f"{x}"', "match x:", "    case 1:", "try:", "finally:"]
        rnd = random.Random(7)
        for _ in range(200):
            text = "\n".join(rnd.choices(pool, k=rnd.randint(5, 32)))
            self.assertEqual(evalplus_sanitize.code_extract(text), upstream(text), repr(text))

    def test_degenerate_prose_response_is_extracted_quickly(self) -> None:
        import time

        from benchmark_utils import evalplus_sanitize

        # Upstream (and the previous pruned version) scan this shape cubically:
        # 480 prose lines already took ~4 s, and a real 16k-token truncated
        # response stalled MBPP+ rescoring for hours.
        prose = ["The answer is:", "It follows that the result holds.",
                 "- see the step above", "1. rewrite the loop as follows"]
        code = "def solve(x):\n    y = x + 1\n    return y * 2"
        rnd = random.Random(1)
        lines = [rnd.choice(prose) for _ in range(2000)]
        lines[1000:1000] = code.split("\n")
        started = time.monotonic()
        self.assertEqual(evalplus_sanitize.code_extract("\n".join(lines)), code)
        self.assertLess(time.monotonic() - started, 30.0)

    def test_parse_budget_keeps_the_best_span_found_so_far(self) -> None:
        import contextlib
        import io
        from unittest import mock

        from benchmark_utils import evalplus_sanitize

        text = "x = 1\ny = 2\nThe answer is:\nz = 3"
        with mock.patch.object(evalplus_sanitize, "_PARSE_BUDGET_CHARS", 1):
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                out = evalplus_sanitize.code_extract(text)
        self.assertEqual(out, "x = 1")  # upstream's default span, nothing confirmed yet
        self.assertIn("parse budget", log.getvalue())
        self.assertEqual(
            evalplus_sanitize.code_extract(text),
            evalplus_sanitize.UPSTREAM_CODE_EXTRACT(text),
        )


if __name__ == "__main__":
    unittest.main()
