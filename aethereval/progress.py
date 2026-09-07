"""Terminal progress bars and newline-based progress for server logs."""

import sys
import time

from tqdm.auto import tqdm


class Progress:
    def __init__(self, total, desc, unit="gen", enabled=True):
        self.total, self.desc, self.unit = total, desc, unit
        self.enabled = enabled
        self.n = 0
        self.started = self.last_log = time.monotonic()
        self.closed = False
        self.bar = (
            tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, mininterval=1.0)
            if enabled and sys.stderr.isatty()
            else None
        )
        if enabled and self.bar is None:
            self.refresh(force=True)

    def update(self, count=1):
        self.n += count
        if self.bar is not None:
            self.bar.update(count)
        else:
            self.refresh()

    def refresh(self, force=False):
        if not self.enabled or self.bar is not None:
            return
        now = time.monotonic()
        if not force and now - self.last_log < 10:
            return
        elapsed = now - self.started
        rate = self.n / elapsed if elapsed > 0 else 0
        eta = f"{max(0, self.total - self.n) / rate:.0f}s" if rate > 0 else "unknown"
        print(
            f"[aethereval] {self.desc}: {self.n}/{self.total} {self.unit} "
            f"rate={rate:.2f}/s elapsed={elapsed:.0f}s ETA={eta}",
            flush=True,
        )
        self.last_log = now

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.bar is not None:
            self.bar.close()
        else:
            self.refresh(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
