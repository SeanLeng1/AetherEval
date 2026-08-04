"""Narrow AetherEval scoring adaptations for BFCL.

BFCL's Python AST checker treats a JSON scalar encoded as a string as a type error,
even when the tool schema makes the intended scalar unambiguous. ToolRL training data
frequently serializes such arguments (for example, ``"2"`` for an integer).  The
adapter below makes only those schema-authorized scalar quotations equivalent. It does
not coerce strings for string/any parameters, containers, malformed values, missing
arguments, or function names.
"""

from __future__ import annotations

import functools
import importlib
import json
import math
from typing import Callable


def _coerce_quoted_scalar(value, declared_type: str):
    if type(value) is not str or declared_type not in {
        "integer",
        "float",
        "boolean",
    }:
        return value

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value

    if declared_type == "integer":
        return parsed if type(parsed) is int else value
    if declared_type == "boolean":
        return parsed if type(parsed) is bool else value

    if type(parsed) not in {int, float} or type(parsed) is bool:
        return value
    parsed = float(parsed)
    return parsed if math.isfinite(parsed) else value


def _normalize_python_calls(
    func_descriptions,
    model_output,
    *,
    name_converter: Callable[[str], str] | None = None,
):
    """Return a non-mutating, schema-aware normalization of decoded Python calls."""
    if not isinstance(model_output, list):
        return model_output

    descriptions = (
        func_descriptions if isinstance(func_descriptions, list) else [func_descriptions]
    )
    descriptions_by_name = {}
    for description in descriptions:
        if not isinstance(description, dict) or "name" not in description:
            continue
        name = description["name"]
        descriptions_by_name[name] = description
        if name_converter is not None:
            descriptions_by_name[name_converter(name)] = description

    normalized_output = []
    for call in model_output:
        if not isinstance(call, dict):
            normalized_output.append(call)
            continue

        normalized_call = call
        for function_name, parameters in call.items():
            description = descriptions_by_name.get(function_name)
            if description is None or not isinstance(parameters, dict):
                continue
            properties = (
                description.get("parameters", {}).get("properties", {})
            )
            normalized_parameters = parameters
            for parameter_name, value in parameters.items():
                parameter_schema = properties.get(parameter_name)
                if not isinstance(parameter_schema, dict):
                    continue
                normalized = _coerce_quoted_scalar(
                    value,
                    str(parameter_schema.get("type", "")),
                )
                if type(normalized) is type(value):
                    continue
                if normalized_parameters is parameters:
                    normalized_parameters = dict(parameters)
                normalized_parameters[parameter_name] = normalized

            if normalized_parameters is not parameters:
                if normalized_call is call:
                    normalized_call = dict(call)
                normalized_call[function_name] = normalized_parameters

        normalized_output.append(normalized_call)
    return normalized_output


def _decoded_calls_to_execute(model_output) -> list[str]:
    calls = []
    for call in model_output:
        if not isinstance(call, dict):
            continue
        for function_name, parameters in call.items():
            if not isinstance(parameters, dict):
                continue
            arguments = ", ".join(
                f"{name}={value!r}" for name, value in parameters.items()
            )
            calls.append(f"{function_name}({arguments})")
    return calls


class _SchemaAwareExecutionHandler:
    """Per-entry proxy adding scalar normalization to multi-turn execution."""

    def __init__(self, handler, func_descriptions):
        self._handler = handler
        self._func_descriptions = func_descriptions

    def __getattr__(self, name):
        return getattr(self._handler, name)

    def decode_execute(self, result, has_tool_call_tag):
        from bfcl_eval.constants.enums import ReturnFormat

        decoded = self._handler.decode_ast(
            result,
            ReturnFormat.PYTHON,
            has_tool_call_tag,
        )
        normalized = _normalize_python_calls(self._func_descriptions, decoded)
        return _decoded_calls_to_execute(normalized)


def install_scalar_quotation_tolerance() -> None:
    """Install the pinned-BFCL hooks once for the current evaluation process."""
    eval_runner = importlib.import_module("bfcl_eval.eval_checker.eval_runner")
    if getattr(eval_runner, "_aethereval_scalar_quotation_tolerance", False):
        return

    ast_module = importlib.import_module(
        "bfcl_eval.eval_checker.ast_eval.ast_checker"
    )
    original_ast_checker = eval_runner.ast_checker

    @functools.wraps(original_ast_checker)
    def ast_checker(
        func_description,
        model_output,
        possible_answer,
        language,
        test_category,
        model_name,
    ):
        if language == ast_module.Language.PYTHON:
            model_output = _normalize_python_calls(
                func_description,
                model_output,
                name_converter=lambda name: ast_module.convert_func_name(
                    name,
                    model_name,
                ),
            )
        return original_ast_checker(
            func_description,
            model_output,
            possible_answer,
            language,
            test_category,
            model_name,
        )

    original_multi_turn_entry = eval_runner._evaluate_single_multi_turn_entry

    @functools.wraps(original_multi_turn_entry)
    def evaluate_single_multi_turn_entry(
        handler,
        test_entry_id,
        model_result_list,
        ground_truth_list,
        prompt_entry,
        model_name,
        test_category,
    ):
        handler = _SchemaAwareExecutionHandler(handler, prompt_entry["function"])
        return original_multi_turn_entry(
            handler,
            test_entry_id,
            model_result_list,
            ground_truth_list,
            prompt_entry,
            model_name,
            test_category,
        )

    eval_runner.ast_checker = ast_checker
    eval_runner._evaluate_single_multi_turn_entry = evaluate_single_multi_turn_entry
    eval_runner._aethereval_scalar_quotation_tolerance = True


__all__ = ["install_scalar_quotation_tolerance"]
