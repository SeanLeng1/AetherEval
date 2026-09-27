import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from benchmark_utils.data import read_text, write_task_jsonl, write_text


class DataHelperTests(unittest.TestCase):
    def test_read_text_checks_the_pinned_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.csv"
            source.write_bytes(b"a,b\n")
            digest = hashlib.sha256(b"a,b\n").hexdigest()

            self.assertEqual(read_text(source, sha256=digest), "a,b\n")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                read_text(source, sha256="0" * 64)

    def test_writers_replace_files_without_leaving_temp_files(self) -> None:
        rows = [{"id": "q1", "text": "café ’"}]
        raw_line = '{"id": "q1", "text": "\\u2019"}'
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp)
            with redirect_stdout(StringIO()):
                write_task_jsonl(task_dir, rows)
            out_path = task_dir / "data" / "eval.jsonl"
            self.assertEqual(
                out_path.read_text(encoding="utf-8"),
                json.dumps(rows[0], ensure_ascii=False) + "\n",
            )

            write_text(out_path, raw_line + "\n")
            # Verbatim: the upstream \u2019 escape is not re-serialized.
            self.assertEqual(out_path.read_text(encoding="utf-8"), raw_line + "\n")
            self.assertEqual(sorted(p.name for p in out_path.parent.iterdir()), ["eval.jsonl"])


if __name__ == "__main__":
    unittest.main()
