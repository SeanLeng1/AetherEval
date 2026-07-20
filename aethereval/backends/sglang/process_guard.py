import ctypes
import os
import signal
import sys


_PR_SET_PDEATHSIG = 1


def _arm_parent_death_signal(expected_parent_pid: int) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return os.getppid() == expected_parent_pid


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: process_guard PARENT_PID COMMAND [ARG ...]")
    expected_parent_pid = int(sys.argv[1])
    if not _arm_parent_death_signal(expected_parent_pid):
        raise SystemExit(1)
    os.execvp(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
