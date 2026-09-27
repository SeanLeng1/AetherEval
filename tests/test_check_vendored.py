import unittest
from pathlib import Path

from aethereval.core.task_register import _load_module_from_path

check_vendored = _load_module_from_path(
    "check_vendored", Path(__file__).resolve().parents[1] / "tools" / "check_vendored.py"
)


class CheckVendoredTests(unittest.TestCase):
    def test_formatting_comments_and_docstrings_are_ignored(self) -> None:
        upstream = (
            '"""Module doc."""\n'
            "class A:\n"
            '\t"""Class doc."""\n'
            "\tdef f(self, x):  # comment\n"
            "\t\t'''Function doc.'''\n"
            "\t\treturn  (x+1)\n"
        )
        local = "class A:\n\n    def f(self, x):\n        return x + 1\n"
        self.assertEqual(check_vendored.normalize(upstream), check_vendored.normalize(local))

    def test_code_changes_remain(self) -> None:
        self.assertNotEqual(
            check_vendored.normalize("def f(x):\n    return x + 1\n"),
            check_vendored.normalize("def f(x):\n    return x + 2\n"),
        )
        # A docstring-only body must stay valid code.
        self.assertEqual(
            check_vendored.normalize('def f():\n    """Doc."""\n'), ["def f():", "    pass"]
        )


if __name__ == "__main__":
    unittest.main()
