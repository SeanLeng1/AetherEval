from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aethereval.io import read_jsonl


class IOTests(unittest.TestCase):
    def test_read_jsonl_detects_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:deadbeef\n"
                "size 123\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                read_jsonl(path)

    def test_read_jsonl_normal_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            rows = [{"id": "1"}, {"id": "2"}]
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            loaded = read_jsonl(path)
            self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
