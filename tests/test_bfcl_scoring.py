import json
import unittest
from unittest import mock

from benchmarks.bfcl._compat import ensure_bfcl_importable

ensure_bfcl_importable()

from benchmarks.bfcl.handler import RLLAHandler  # noqa: E402
from benchmarks.bfcl.register import register_rlla_model  # noqa: E402
from benchmarks.bfcl.scoring import (  # noqa: E402
    _normalize_python_calls,
    install_scalar_quotation_tolerance,
)


def _tool_schema():
    return {
        "name": "typed_tool",
        "description": "A tool with several parameter types.",
        "parameters": {
            "type": "dict",
            "properties": {
                "count": {"type": "integer"},
                "ratio": {"type": "float"},
                "enabled": {"type": "boolean"},
                "label": {"type": "string"},
                "items": {"type": "array", "items": {"type": "integer"}},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "dict",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
                "options": {
                    "type": "dict",
                    "properties": {"threshold": {"type": "float"}},
                },
            },
            "required": ["count", "ratio", "enabled", "label", "items"],
        },
    }


class BfclScoringTests(unittest.TestCase):
    def test_normalization_is_narrow_schema_aware_and_non_mutating(self) -> None:
        output = [
            {
                "typed_tool": {
                    "count": "2",
                    "ratio": "2",
                    "enabled": "true",
                    "label": "2",
                    "items": "[1, 2]",
                    "filters": [{"enabled": "false", "limit": "3"}],
                    "options": {"threshold": "0.5", "unknown": "true"},
                }
            }
        ]
        original = json.loads(json.dumps(output))

        normalized = _normalize_python_calls([_tool_schema()], output)

        self.assertEqual(output, original)
        self.assertEqual(normalized[0]["typed_tool"]["count"], 2)
        self.assertEqual(type(normalized[0]["typed_tool"]["count"]), int)
        self.assertEqual(normalized[0]["typed_tool"]["ratio"], 2.0)
        self.assertEqual(type(normalized[0]["typed_tool"]["ratio"]), float)
        self.assertIs(normalized[0]["typed_tool"]["enabled"], True)
        self.assertEqual(normalized[0]["typed_tool"]["label"], "2")
        self.assertEqual(normalized[0]["typed_tool"]["items"], "[1, 2]")
        self.assertEqual(
            normalized[0]["typed_tool"]["filters"],
            [{"enabled": False, "limit": 3}],
        )
        self.assertEqual(
            normalized[0]["typed_tool"]["options"],
            {"threshold": 0.5, "unknown": "true"},
        )

    def test_invalid_or_ambiguous_scalar_strings_remain_strict(self) -> None:
        output = [
            {
                "typed_tool": {
                    "count": "2.0",
                    "ratio": "NaN",
                    "enabled": "True",
                }
            }
        ]

        normalized = _normalize_python_calls([_tool_schema()], output)

        self.assertEqual(normalized, output)

    def test_installed_ast_checker_accepts_equivalent_quoted_scalars(self) -> None:
        registry_name = "aethereval-quotation-test"
        register_rlla_model(registry_name)
        install_scalar_quotation_tolerance()

        from bfcl_eval.eval_checker import eval_runner

        first_checker = eval_runner.ast_checker
        install_scalar_quotation_tolerance()
        self.assertIs(eval_runner.ast_checker, first_checker)

        result = eval_runner.ast_checker(
            [_tool_schema()],
            [
                {
                    "typed_tool": {
                        "count": "2",
                        "ratio": "2.5",
                        "enabled": "false",
                        "label": "2",
                        "items": [1, 2],
                    }
                }
            ],
            [
                {
                    "typed_tool": {
                        "count": [2],
                        "ratio": [2.5],
                        "enabled": [False],
                        "label": ["2"],
                        "items": [[1, 2]],
                    }
                }
            ],
            "Python",
            "simple",
            registry_name,
        )

        self.assertIs(result["valid"], True)

    def test_installed_ast_checker_keeps_encoded_containers_strict(self) -> None:
        registry_name = "aethereval-quotation-container-test"
        register_rlla_model(registry_name)
        install_scalar_quotation_tolerance()

        from bfcl_eval.eval_checker import eval_runner

        result = eval_runner.ast_checker(
            [_tool_schema()],
            [
                {
                    "typed_tool": {
                        "count": 2,
                        "ratio": 2.5,
                        "enabled": False,
                        "label": "2",
                        "items": "[1, 2]",
                    }
                }
            ],
            [
                {
                    "typed_tool": {
                        "count": [2],
                        "ratio": [2.5],
                        "enabled": [False],
                        "label": ["2"],
                        "items": [[1, 2]],
                    }
                }
            ],
            "Python",
            "simple",
            registry_name,
        )

        self.assertIs(result["valid"], False)
        self.assertEqual(result["error_type"], "type_error:simple")

    def test_multi_turn_execution_uses_same_schema_aware_rule(self) -> None:
        handler = RLLAHandler.__new__(RLLAHandler)
        result = (
            "<think>Call the tool.</think>\n<tool_call>\n"
            '{"name":"typed_tool","parameters":{"count":"2",'
            '"ratio":"2.5","enabled":"false","label":"2",'
            '"items":[1,2]}}\n</tool_call>'
        )

        with mock.patch(
            "benchmarks.bfcl.scoring._multi_turn_descriptions_by_name",
            return_value={"typed_tool": _tool_schema()},
        ):
            calls = handler.decode_execute(result)

        self.assertEqual(
            calls,
            ["typed_tool(count=2, ratio=2.5, enabled=False, label='2', items=[1, 2])"],
        )


if __name__ == "__main__":
    unittest.main()
