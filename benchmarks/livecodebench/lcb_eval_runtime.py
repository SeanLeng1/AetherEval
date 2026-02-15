from __future__ import annotations

import ast
import faulthandler
import json
import multiprocessing
import platform
import signal
import sys
import time
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from typing import Any
from unittest.mock import mock_open, patch


try:
    sys.set_int_max_str_digits(50000)
except Exception:
    pass


_IMPORT_STRING = (
    "from string import *\n"
    "from re import *\n"
    "from datetime import *\n"
    "from collections import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from copy import *\n"
    "from math import *\n"
    "from random import *\n"
    "from statistics import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from operator import *\n"
    "from io import *\n"
    "from sys import *\n"
    "from json import *\n"
    "from builtins import *\n"
    "from typing import *\n"
    "import string\n"
    "import re\n"
    "import datetime\n"
    "import collections\n"
    "import heapq\n"
    "import bisect\n"
    "import copy\n"
    "import math\n"
    "import random\n"
    "import statistics\n"
    "import itertools\n"
    "import functools\n"
    "import operator\n"
    "import io\n"
    "import sys\n"
    "import json\n"
    "sys.setrecursionlimit(50000)\n"
)


class CodeType(Enum):
    CALL_BASED = 0
    STANDARD_INPUT = 1


class TimeoutException(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutException("alarm timeout")


class Capturing(list):
    def __enter__(self) -> "Capturing":
        self._stdout = sys.stdout
        self._stringio = StringIO()
        # Keep compatibility with code that closes stdout.
        self._stringio.close = lambda: None  # type: ignore[assignment]
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args: Any) -> None:
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class _MockBuffer:
    def __init__(self, inputs: str) -> None:
        self._inputs = inputs.encode("utf-8")

    def read(self, *args: Any) -> bytes:
        return self._inputs

    def readline(self, *args: Any) -> bytes:
        return self._inputs.split(b"\n")[0] + b"\n"


class _MockStdinWithBuffer:
    def __init__(self, inputs: str) -> None:
        self._inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = _MockBuffer(inputs)

    def read(self, *args: Any) -> str:
        return self._inputs

    def readline(self, *args: Any) -> str:
        return self._stringio.readline(*args)

    def readlines(self, *args: Any) -> list[str]:
        return self._inputs.split("\n")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stringio, name)


def _clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = ast.unparse(last_block.test).strip()
            if condition == "__name__ == '__main__'":
                prefix = "\n".join(ast.unparse(stmt) for stmt in astree.body[:-1])
                body = "\n".join(ast.unparse(stmt) for stmt in last_block.body)
                code = prefix + ("\n" if prefix else "") + body
    except Exception:
        return code
    return code


def _make_function(code: str) -> str:
    try:
        astree = ast.parse(code)
        import_stmts = []
        body_stmts = []
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                body_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=body_stmts,
            decorator_list=[],
            lineno=-1,
        )
        import_text = "\n".join(ast.unparse(stmt) for stmt in import_stmts)
        return (
            _IMPORT_STRING
            + "\n"
            + import_text
            + "\n"
            + ast.unparse(function_ast)
        )
    except Exception:
        return code


def _call_method(method: Any, inputs: list[str] | str) -> Any:
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    iter_lines = iter(inputs.split("\n"))
    mock_stdin = _MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(iter_lines))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call(inner_method: Any) -> Any:
        try:
            return inner_method()
        except SystemExit:
            return None

    return _inner_call(method)


def _get_function(compiled_sol: Any, fn_name: str | None) -> Any:
    if not fn_name:
        return None
    try:
        if not hasattr(compiled_sol, fn_name):
            return None
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def _compile_code(code: str, timeout: int) -> Any:
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            return tmp_sol.Solution()
        return tmp_sol
    finally:
        signal.alarm(0)


def _convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        return True, [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []


def _get_stripped_lines(value: str) -> list[str]:
    text = value.strip()
    return [item.strip() for item in text.split("\n")]


def _grade_call_based(
    code: str,
    all_inputs: list[str],
    all_outputs: list[str],
    fn_name: str,
    timeout: int,
) -> list[int | bool]:
    code = _IMPORT_STRING + "\n\n" + code
    compiled_sol = _compile_code(code, timeout)
    if compiled_sol is None:
        return [-4]

    method = _get_function(compiled_sol, fn_name)
    if method is None:
        return [-4]

    parsed_inputs = [
        [json.loads(line) for line in inp.split("\n") if line.strip()]
        for inp in all_inputs
    ]
    parsed_outputs = [json.loads(out) for out in all_outputs]

    results: list[int | bool] = []
    for gt_input, gt_output in zip(parsed_inputs, parsed_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            prediction = method(*gt_input)
            signal.alarm(0)
            if isinstance(prediction, tuple):
                prediction = list(prediction)
            is_correct = prediction == gt_output
            results.append(bool(is_correct))
            if not is_correct:
                return results
        except Exception as exc:
            signal.alarm(0)
            if "timeoutexception" in repr(exc).lower():
                results.append(-3)
            else:
                results.append(-4)
            return results
        finally:
            signal.alarm(0)
            faulthandler.disable()

    return results


def _grade_stdio(
    code: str,
    all_inputs: list[str],
    all_outputs: list[str],
    timeout: int,
) -> list[int | bool]:
    code = _clean_if_name(code)
    code = _make_function(code)

    compiled_sol = _compile_code(code, timeout)
    if compiled_sol is None:
        return [-4]

    method = _get_function(compiled_sol, "wrapped_function")
    if method is None:
        return [-4]

    results: list[int | bool] = []
    for gt_input, gt_output in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with Capturing() as captured_output:
            try:
                _call_method(method, gt_input)
                signal.alarm(0)
            except Exception as exc:
                signal.alarm(0)
                if "timeoutexception" in repr(exc).lower():
                    results.append(-3)
                else:
                    results.append(-4)
                return results
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0] if captured_output else ""
        pred_lines = _get_stripped_lines(prediction)
        gold_lines = _get_stripped_lines(gt_output)

        if len(pred_lines) != len(gold_lines):
            results.append(-2)
            return results

        matched = True
        for pred_line, gold_line in zip(pred_lines, gold_lines):
            if pred_line == gold_line:
                continue

            ok_pred, pred_decimals = _convert_line_to_decimals(pred_line)
            if not ok_pred:
                matched = False
                break
            ok_gold, gold_decimals = _convert_line_to_decimals(gold_line)
            if not ok_gold:
                matched = False
                break
            if pred_decimals != gold_decimals:
                matched = False
                break

        if matched:
            results.append(True)
        else:
            results.append(-2)
            return results

    return results


def _safe_disable_attr(obj: Any, name: str) -> None:
    if hasattr(obj, name):
        try:
            setattr(obj, name, None)
        except Exception:
            return


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    import os
    import shutil
    import subprocess

    os.environ["OMP_NUM_THREADS"] = "1"

    _safe_disable_attr(builtins, "quit")

    for attr in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ):
        _safe_disable_attr(os, attr)

    for attr in ("rmtree", "move", "chown"):
        _safe_disable_attr(shutil, attr)

    _safe_disable_attr(subprocess, "Popen")
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        try:
            sys.modules[name] = None
        except Exception:
            continue

    try:
        if isinstance(__builtins__, dict):
            __builtins__["help"] = None
    except Exception:
        pass


def run_test(sample: dict[str, Any], test: str | None = None, timeout: int = 6) -> list[int | bool]:
    signal.signal(signal.SIGALRM, _timeout_handler)
    reliability_guard()

    in_outs = json.loads(sample["input_output"])
    fn_name = in_outs.get("fn_name")
    code_type = CodeType.CALL_BASED if fn_name is not None else CodeType.STANDARD_INPUT

    if test is None:
        return [-4]

    if code_type == CodeType.CALL_BASED:
        signal.alarm(timeout)
        try:
            return _grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=str(fn_name),
                timeout=timeout,
            )
        except Exception:
            return [-4]
        finally:
            signal.alarm(0)

    signal.alarm(timeout)
    try:
        return _grade_stdio(
            code=test,
            all_inputs=in_outs["inputs"],
            all_outputs=in_outs["outputs"],
            timeout=timeout,
        )
    except Exception:
        return [-4]
    finally:
        signal.alarm(0)


def check_correctness(sample: dict[str, Any], generation: str, timeout: int) -> list[int | bool]:
    def _temp_run(local_sample: dict[str, Any], local_generation: str, result: Any) -> None:
        result.append(run_test(local_sample, test=local_generation, timeout=timeout))

    manager = multiprocessing.Manager()
    result = manager.list()
    process = multiprocessing.Process(target=_temp_run, args=(sample, generation, result))
    process.start()
    process.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
    if process.is_alive():
        process.kill()

    if not result:
        num_tests = len(json.loads(sample["input_output"])["inputs"])
        return [-1 for _ in range(num_tests)]
    return list(result[0])


def evaluate_candidate(
    *,
    code: str,
    inputs: list[str],
    outputs: list[str],
    fn_name: str | None,
    timeout: int = 6,
) -> tuple[list[int | bool], str | None]:
    payload = {
        "input_output": json.dumps(
            {
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": fn_name,
            }
        )
    }
    start = time.time()
    try:
        statuses = check_correctness(payload, code, timeout=int(timeout))
    except Exception as exc:  # noqa: BLE001
        return [-5], f"{type(exc).__name__}: {exc}"

    elapsed = time.time() - start
    if elapsed > max(2.0, (timeout + 1) * max(len(inputs), 1) + 5.0):
        return statuses, "global timeout"
    return statuses, None
