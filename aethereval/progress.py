"""Shared tqdm bars for preprocessing, generation and scoring."""

import sys
from time import monotonic

from tqdm.auto import tqdm


class Progress(tqdm):
    def __init__(self, total, desc, unit="gen", enabled=True):
        self._terminal = sys.stderr.isatty()
        self._last_snapshot = None
        self._closing = False
        super().__init__(
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stderr,
            dynamic_ncols=self._terminal,
            mininterval=1.0 if self._terminal else 10.0,
            miniters=1,
            disable=not enabled,
        )

    def display(self, msg=None, pos=None):
        if self._terminal:
            return super().display(msg, pos)
        now = monotonic()
        # Explicit refreshes also respect the cloud-log interval; close always emits the final state.
        if not self._closing and self._last_snapshot is not None and now - self._last_snapshot < 10.0:
            return False
        self.fp.write((str(self) if msg is None else msg) + "\n")
        self.fp.flush()
        self._last_snapshot = now
        return True

    def close(self):
        self._closing = True
        super().close()
