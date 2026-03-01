
import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import re
import signal
import tempfile
from multiprocessing import Value
from typing import Any

from aethereval.metrics.common import aggregate_binary_results, mean, mean_stderr, to_records
from aethereval.core.types import GenerationRecord, Sample

PRIMARY_METRIC = "pass@1"


_CODE_BLOCK_RE = re.compile(r"```(?:python)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(
    r"here is the completed function:\s*```(?:python)?[ \t]*\n?(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_DEFAULT_TIMEOUT_SEC = 20.0
_PROCESS_TERMINATE_GRACE_SEC = 1.0
_PROCESS_KILL_GRACE_SEC = 1.0


def _empty_aggregate_result() -> dict[str, float]:
    return {
        "accuracy": 0.0,
        "accuracy_stderr": 0.0,
        "accuracy_plus": 0.0,
        "accuracy_plus_stderr": 0.0,
        "accuracy_base": 0.0,
        "accuracy_base_stderr": 0.0,
        "pass@1": 0.0,
        "pass@1_stderr": 0.0,
    }


class _TimeoutException(Exception):
    pass


def _record_plus_score(record: GenerationRecord) -> float:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    plus_pass = bool(parsed.get("plus_pass", bool(record.score >= 1.0)))
    return 1.0 if plus_pass else 0.0


def _record_base_score(record: GenerationRecord) -> float:
    parsed = record.parsed if isinstance(record.parsed, dict) else {}
    plus_pass = bool(parsed.get("plus_pass", bool(record.score >= 1.0)))
    base_pass = bool(parsed.get("base_pass", plus_pass))
    return 1.0 if base_pass else 0.0


def _record_has_parsed(record: GenerationRecord) -> bool:
    return isinstance(record.parsed, dict) and bool(record.parsed)


@contextlib.contextmanager
def _time_limit(seconds: float):
    def _signal_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise _TimeoutException("Timed out.")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class _WriteOnlyStringIO(io.StringIO):
    def read(self, *args: Any, **kwargs: Any) -> str:
        raise OSError

    def readline(self, *args: Any, **kwargs: Any) -> str:
        raise OSError

    def readlines(self, *args: Any, **kwargs: Any) -> list[str]:
        raise OSError

    def readable(self, *args: Any, **kwargs: Any) -> bool:
        return False


class _redirect_stdin(contextlib._RedirectStream):  # type: ignore[misc]
    _stream = "stdin"


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield


@contextlib.contextmanager
def _chdir(path: str):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


def _reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None  # type: ignore[assignment]
    os.system = None  # type: ignore[assignment]
    os.putenv = None  # type: ignore[assignment]
    os.remove = None  # type: ignore[assignment]
    os.removedirs = None  # type: ignore[assignment]
    os.rmdir = None  # type: ignore[assignment]
    os.fchdir = None  # type: ignore[assignment]
    os.setuid = None  # type: ignore[assignment]
    os.fork = None  # type: ignore[assignment]
    os.forkpty = None  # type: ignore[assignment]
    os.killpg = None  # type: ignore[assignment]
    os.rename = None  # type: ignore[assignment]
    os.renames = None  # type: ignore[assignment]
    os.truncate = None  # type: ignore[assignment]
    os.replace = None  # type: ignore[assignment]
    os.unlink = None  # type: ignore[assignment]
    os.fchmod = None  # type: ignore[assignment]
    os.fchown = None  # type: ignore[assignment]
    os.chmod = None  # type: ignore[assignment]
    os.chown = None  # type: ignore[assignment]
    os.chroot = None  # type: ignore[assignment]
    os.lchflags = None  # type: ignore[assignment]
    os.lchmod = None  # type: ignore[assignment]
    os.lchown = None  # type: ignore[assignment]
    os.getcwd = None  # type: ignore[assignment]
    os.chdir = None  # type: ignore[assignment]

    import shutil

    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]


def _unsafe_execute(result: Value, solution: str, test_code: str, timeout_sec: float) -> None:
    with _create_tempdir():
        import os as _os
        import shutil as _shutil

        orig = {
            "rmtree": _shutil.rmtree,
            "rmdir": _os.rmdir,
            "chdir": _os.chdir,
            "unlink": _os.unlink,
        }
        _reliability_guard()
        check_program = solution + "\n" + test_code

        try:
            exec_globals: dict[str, Any] = {}
            with _swallow_io():
                with _time_limit(timeout_sec):
                    exec(check_program, exec_globals)  # noqa: S102
            result.value = 1
        except _TimeoutException:
            result.value = -1
        except BaseException:
            result.value = 0
        finally:
            _shutil.rmtree = orig["rmtree"]
            _os.rmdir = orig["rmdir"]
            _os.chdir = orig["chdir"]
            _os.unlink = orig["unlink"]


def _terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.01)
        return

    process.terminate()
    process.join(timeout=_PROCESS_TERMINATE_GRACE_SEC)

    if process.is_alive():
        process.kill()
        process.join(timeout=_PROCESS_KILL_GRACE_SEC)


def _run_olmes_style_check(solution: str, test_code: str, timeout_sec: float) -> tuple[bool, str]:
    if not test_code.strip():
        return False, "missing_test"

    result = Value("i", 0)
    process = multiprocessing.Process(
        target=_unsafe_execute,
        args=(result, solution, test_code, timeout_sec),
    )
    process.start()
    process.join(timeout=timeout_sec + 1.0)
    _terminate_process(process)

    if result.value == 1:
        return True, "passed"
    if result.value == -1:
        return False, "timed out"
    return False, "failed"


def _strip_fence_wrapper(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text.rstrip()

    lines = stripped.splitlines()
    if not lines:
        return ""

    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).rstrip()


def _extract_python_candidate(text: str) -> str:
    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        return answer_match.group(1).rstrip()

    blocks = [m.group(1).rstrip() for m in _CODE_BLOCK_RE.finditer(text)]
    if blocks:
        # Prefer the final block to match common "reasoning + final code block" outputs.
        return blocks[-1]

    lower = text.lower()
    marker = "here is the completed function:"
    marker_idx = lower.find(marker)
    if marker_idx >= 0:
        return text[marker_idx + len(marker) :].rstrip()

    return text.rstrip()


def _candidate_solution(sample: Sample, generation: str) -> tuple[str, bool]:
    prompt = str(sample.data["prompt"])
    entry_point = str(sample.data["entry_point"])

    extracted = _extract_python_candidate(generation)
    extracted = _strip_fence_wrapper(extracted)
    if not extracted.strip():
        return prompt, False

    continuation = extracted
    is_full_solution = False
    if continuation.startswith(prompt):
        continuation = continuation[len(prompt) :]
        is_full_solution = True
    elif prompt in continuation:
        continuation = continuation.split(prompt, 1)[1]
        is_full_solution = True
    elif re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", continuation):
        is_full_solution = True

    continuation = continuation.lstrip("\n")
    if not continuation:
        return prompt, is_full_solution

    joiner = "" if prompt.endswith("\n") else "\n"
    # Match OLMES HumanEval behavior: always execute prompt + continuation.
    return prompt + joiner + continuation, is_full_solution


def score_generation(sample: Sample, generation: str) -> dict[str, Any]:
    entry_point = str(sample.data["entry_point"])
    solution, is_full_solution = _candidate_solution(sample, generation)
    test_body = str(sample.data.get("test", "")).rstrip()
    check_code = test_body
    if check_code:
        check_code = f"{check_code}\ncheck({entry_point})"

    timeout_sec = max(0.1, float(sample.data.get("timeout", _DEFAULT_TIMEOUT_SEC)))
    passed, exec_result = _run_olmes_style_check(solution, check_code, timeout_sec)

    parsed = {
        "base_status": exec_result,
        "plus_status": exec_result,
        "base_pass": passed,
        "plus_pass": passed,
        "exec_result": exec_result,
    }
    return {
        "score": 1.0 if passed else 0.0,
        "is_pass": passed,
        "parsed": parsed,
        "meta": {
            "base_pass": passed,
            "plus_pass": passed,
            "exec_result": exec_result,
            "full_solution": is_full_solution,
        },
    }


def aggregate(
    sample_results: list[dict[str, Any]],
    metric_options: dict[str, Any] | None = None,
) -> dict[str, float]:
    if not sample_results:
        return _empty_aggregate_result()

    base_means: list[float] = []
    plus_metrics = aggregate_binary_results(
        sample_results,
        metric_options,
        score_fn=_record_plus_score,
        parsed_flag_fn=_record_has_parsed,
    )

    for item in sample_results:
        records = to_records(item.get("records", []))
        if not records:
            continue

        base_scores = [_record_base_score(record) for record in records]
        base_means.append(mean(base_scores))

    if not base_means:
        return _empty_aggregate_result()

    accuracy_plus = float(plus_metrics.get("accuracy", 0.0))
    accuracy_plus_stderr = float(plus_metrics.get("accuracy_stderr", 0.0))
    result: dict[str, float] = dict(plus_metrics)
    result.update(
        {
            "accuracy": accuracy_plus,
            "accuracy_stderr": accuracy_plus_stderr,
            "accuracy_plus": accuracy_plus,
            "accuracy_plus_stderr": accuracy_plus_stderr,
            "accuracy_base": mean(base_means),
            "accuracy_base_stderr": mean_stderr(base_means),
        }
    )

    return result
