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


if __name__ == "__main__":
    unittest.main()
