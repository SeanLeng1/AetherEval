import json
import tempfile
import unittest
from pathlib import Path

from aethereval.core.io import read_jsonl, write_json


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

    def test_read_jsonl_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_jsonl(Path(tmp) / "missing.jsonl")

    def test_write_json_replaces_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            write_json(path, {"score": 1.0, "name": "é"})
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                json.dumps({"score": 1.0, "name": "é"}, indent=2, ensure_ascii=False),
            )
            with self.assertRaises(TypeError):
                write_json(path, {"score": object()})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["score"], 1.0)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["summary.json"])


if __name__ == "__main__":
    unittest.main()
