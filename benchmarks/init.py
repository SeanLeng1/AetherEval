"""Rebuild benchmark data using each task's prepare_data.py (no inference)."""

import argparse
import subprocess
import sys
from pathlib import Path


def process_data(tasks: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent
    available = sorted(path.parent.name for path in root.glob("*/prepare_data.py"))
    selected = tasks or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"No data preparation script for: {', '.join(unknown)}")
    if not tasks:
        print("BFCL uses its official external runner; no local data to rebuild.", flush=True)
    for index, name in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] Preparing {name}", flush=True)
        subprocess.run(
            [sys.executable, "-u", "-m", f"benchmarks.{name}.prepare_data"],
            cwd=root.parent,
            check=True,
        )
    print(f"Rebuilt data for {len(selected)} benchmarks.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="*", help="Benchmark names; default: all native benchmarks")
    process_data(parser.parse_args().tasks)
