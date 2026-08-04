"""Narrow, schema-aware scoring adaptations for BFCL V3.

ToolRL data can serialize a typed JSON scalar as a string, such as ``"2"`` for
an integer or ``"true"`` for a boolean. BFCL correctly checks the declared
parameter type, but that representation mismatch would otherwise become a
false negative. This module only normalizes canonical JSON scalars when the
tool schema explicitly declares the target type.
"""

import functools
import importlib
import json
import math
from collections.abc import Callable


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


def _normalize_schema_value(value, schema):
    """Return a non-mutating normalization authorized by one BFCL schema."""
    if not isinstance(schema, dict):
        return value

    declared_type = schema.get("type")
    if declared_type in {"integer", "float", "boolean"}:
        return _coerce_quoted_scalar(value, declared_type)

    if declared_type in {"array", "tuple"} and isinstance(value, (list, tuple)):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        normalized = [_normalize_schema_value(item, item_schema) for item in value]
        return tuple(normalized) if isinstance(value, tuple) else normalized

    if declared_type in {"dict", "object"} and isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        return {
            key: _normalize_schema_value(item, properties.get(key))
            for key, item in value.items()
        }

    return value


def _descriptions_by_name(
    func_descriptions,
    name_converter: Callable[[str], str] | None = None,
) -> dict[str, dict]:
    descriptions = (
        func_descriptions
        if isinstance(func_descriptions, list)
        else [func_descriptions]
    )
    result = {}
    for description in descriptions:
        if not isinstance(description, dict) or not isinstance(
            description.get("name"), str
        ):
            continue
        name = description["name"]
        result[name] = description
        if name_converter is not None:
            result[name_converter(name)] = description
    return result


def _normalize_parameters(parameters, description):
    if not isinstance(parameters, dict) or not isinstance(description, dict):
        return parameters
    parameter_schema = description.get("parameters")
    if not isinstance(parameter_schema, dict):
        return parameters
    properties = parameter_schema.get("properties")
    if not isinstance(properties, dict):
        return parameters
    return {
        name: _normalize_schema_value(value, properties.get(name))
        for name, value in parameters.items()
    }


def _normalize_python_calls(
    func_descriptions,
    model_output,
    *,
    name_converter: Callable[[str], str] | None = None,
):
    """Normalize BFCL's decoded ``[{function: parameters}]`` representation."""
    if not isinstance(model_output, list):
        return model_output

    descriptions = _descriptions_by_name(func_descriptions, name_converter)
    normalized_output = []
    for call in model_output:
        if not isinstance(call, dict):
            normalized_output.append(call)
            continue
        normalized_output.append(
            {
                function_name: _normalize_parameters(
                    parameters,
                    descriptions.get(function_name),
                )
                for function_name, parameters in call.items()
            }
        )
    return normalized_output


@functools.lru_cache(maxsize=1)
def _multi_turn_descriptions_by_name() -> dict[str, dict]:
    from bfcl_eval.constants.category_mapping import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
    from bfcl_eval.utils import load_file

    descriptions = {}
    for filename in MULTI_TURN_FUNC_DOC_FILE_MAPPING.values():
        for description in load_file(MULTI_TURN_FUNC_DOC_PATH / filename):
            name = description.get("name")
            if not isinstance(name, str):
                continue
            if name in descriptions:
                raise RuntimeError(
                    "BFCL V3 multi-turn schemas contain a duplicate function "
                    f"name: {name!r}."
                )
            descriptions[name] = description
    return descriptions


def normalize_multi_turn_tool_calls(calls):
    """Normalize handler ``[{name, parameters}]`` calls using V3 function docs."""
    if not isinstance(calls, list):
        return calls
    descriptions = _multi_turn_descriptions_by_name()
    normalized = []
    for call in calls:
        if not isinstance(call, dict):
            normalized.append(call)
            continue
        item = dict(call)
        item["parameters"] = _normalize_parameters(
            call.get("parameters", {}),
            descriptions.get(call.get("name")),
        )
        normalized.append(item)
    return normalized


def install_scalar_quotation_tolerance() -> None:
    """Install the BFCL V3 single-turn checker hook once per process."""
    eval_runner = importlib.import_module("bfcl_eval.eval_checker.eval_runner")
    if getattr(eval_runner, "_aethereval_scalar_quotation_tolerance", False):
        return

    ast_module = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker")
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
        if language == "Python":
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

    eval_runner.ast_checker = ast_checker
    eval_runner._aethereval_scalar_quotation_tolerance = True


__all__ = ["install_scalar_quotation_tolerance"]
