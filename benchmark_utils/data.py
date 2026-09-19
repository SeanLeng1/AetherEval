"""Read preparation inputs from a local file or an upstream URL."""

from pathlib import Path
from urllib.request import urlopen


def read_bytes(source: str | Path) -> bytes:
    if str(source).startswith(("http://", "https://")):
        with urlopen(str(source), timeout=120) as response:
            return response.read()
    return Path(source).read_bytes()


def read_text(source: str | Path) -> str:
    return read_bytes(source).decode("utf-8")
