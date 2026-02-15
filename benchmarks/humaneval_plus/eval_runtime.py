from __future__ import annotations

import contextlib
import faulthandler
import io
import math
import multiprocessing
import os
import platform
import signal
import tempfile
import time
from multiprocessing import Array, Value
from typing import Any, Optional

import numpy as np
import psutil


DEFAULT_GT_TIME_LIMIT_FACTOR = 4.0
DEFAULT_MIN_TIME_LIMIT = 4.0

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_MAPPING = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


# Humaneval/032 special oracle
def _poly(xs: list[float], x: float) -> float:
    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))


def trusted_exec(
    code: str,
    inputs: list[Any],
    entry_point: str,
    *,
    record_time: bool = False,
) -> tuple[list[Any], list[float]] | list[Any]:
    exec_globals: dict[str, Any] = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]

    results: list[Any] = []
    times: list[float] = []
    for inp in inputs:
        args = inp if isinstance(inp, (list, tuple)) else [inp]
        if record_time:
            start = time.time()
            results.append(fn(*args))
            times.append(time.time() - start)
        else:
            results.append(fn(*args))

    if record_time:
        return results, times
    return results


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise IOError

    def readline(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise IOError

    def readlines(self, *args: Any, **kwargs: Any) -> list[str]:  # pragma: no cover
        raise IOError

    def readable(self, *args: Any, **kwargs: Any) -> bool:  # pragma: no cover
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore[misc]
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum: int, frame: Any) -> None:  # pragma: no cover
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root: str):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


def query_maximum_memory_bytes() -> Optional[int]:
    maximum_memory_bytes = int(
        os.getenv("EVALPLUS_MAX_MEMORY_BYTES", str(4 * 1024 * 1024 * 1024))
    )
    maximum_memory_bytes = min(maximum_memory_bytes, psutil.virtual_memory().total)
    if maximum_memory_bytes == -1:
        return None
    return maximum_memory_bytes


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    if maximum_memory_bytes is not None:
        system = platform.uname().system
        if system != "Windows":
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if system != "Darwin":
                resource.setrlimit(
                    resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
                )

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import shutil
    import subprocess
    import sys

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None
    builtins.open = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None  # type: ignore[attr-defined]

    try:
        builtins.help = None  # type: ignore[assignment]
    except Exception:  # pragma: no cover
        pass

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _is_floats(x: Any) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype in (np.float64, np.float32)
    return False


def _unsafe_execute(
    entry_point: str,
    code: str,
    inputs: list[Any],
    expected: list[Any],
    time_limits: list[float],
    atol: float,
    fast_check: bool,
    stat: Value,
    details: Array,
    progress: Value,
):
    with create_tempdir():
        import os as _os
        import shutil as _shutil

        rmtree = _shutil.rmtree
        rmdir = _os.rmdir
        chdir = _os.chdir
        reliability_guard(maximum_memory_bytes=query_maximum_memory_bytes())
        exec_globals: dict[str, Any] = {}

        try:
            with swallow_io():
                exec(code, exec_globals)
                fn = exec_globals[entry_point]

            for i, inp in enumerate(inputs):
                args = inp if isinstance(inp, (list, tuple)) else [inp]
                try:
                    with time_limit(time_limits[i]):
                        with swallow_io():
                            out = fn(*args)

                    exp = expected[i]
                    exact_match = out == exp

                    if "find_zero" == entry_point:
                        assert abs(_poly(*args, out)) <= atol
                        details[i] = True
                        progress.value += 1
                        continue

                    if atol == 0 and _is_floats(exp):
                        atol = 1e-6

                    if not exact_match and atol != 0:
                        assert type(out) == type(exp)
                        if isinstance(exp, (list, tuple)):
                            assert len(out) == len(exp)
                        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                    else:
                        assert exact_match
                except BaseException:
                    details[i] = False
                    progress.value += 1
                    if fast_check:
                        raise
                    continue

                details[i] = True
                progress.value += 1

            stat.value = _SUCCESS
        except BaseException:
            stat.value = _FAILED

        _shutil.rmtree = rmtree
        _os.rmdir = rmdir
        _os.chdir = chdir


def untrusted_check(
    code: str,
    inputs: list[Any],
    entry_point: str,
    *,
    expected: list[Any],
    atol: float,
    ref_time: list[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> tuple[str, list[bool]]:
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout_cap = float(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", "60"))
    timeout = min(timeout_cap, sum(time_limits)) + 1
    if not fast_check:
        timeout += 1

    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            float(atol),
            fast_check,
            stat,
            details,
            progress,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    status = _MAPPING.get(int(stat.value))
    done_details = list(details[: int(progress.value)])

    if not status:
        status = TIMEOUT

    if status == PASS:
        if len(done_details) != len(inputs) or not all(done_details):
            status = FAIL

    return status, [bool(x) for x in done_details]
