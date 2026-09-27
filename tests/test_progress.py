import io
import unittest
from unittest import mock

from aethereval.progress import Progress


class ProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        # Progress must never write to stdout; every test checks this.
        self.stdout = io.StringIO()
        patcher = mock.patch("sys.stdout", self.stdout)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_terminal_refreshes_the_same_line(self) -> None:
        stream = io.StringIO()
        with (
            mock.patch("sys.stderr", stream),
            mock.patch.object(stream, "isatty", return_value=True),
        ):
            with Progress(10, "preparing", "prompt") as progress:
                progress.update(5)
                progress.refresh()
                self.assertTrue("50%" in stream.getvalue() and "5/10" in stream.getvalue())
                self.assertNotIn("\n", stream.getvalue())
                progress.update(5)
        output = stream.getvalue()
        self.assertTrue("preparing" in output and "100%" in output and "10/10" in output)
        self.assertTrue("\r" in output and "[aethereval]" not in output)
        self.assertEqual(self.stdout.getvalue(), "")

    def test_nonterminal_emits_flushed_bar_snapshots_without_carriage_returns(self) -> None:
        stream = io.StringIO()
        with (
            mock.patch("sys.stderr", stream),
            mock.patch("aethereval.progress.monotonic") as clock,
            mock.patch.object(stream, "flush", wraps=stream.flush) as flush,
        ):
            clock.return_value = 0.0
            with Progress(10, "sglang generating") as progress:
                initial = stream.getvalue()
                self.assertTrue("0/10" in initial and initial.endswith("\n"))
                progress.update(5)
                clock.return_value = 9.0
                progress.refresh()
                self.assertEqual(stream.getvalue(), initial)
                clock.return_value = 10.0
                progress.refresh()
                self.assertIn("5/10", stream.getvalue())
                snapshot = stream.getvalue()
                clock.return_value = 11.0
                progress.refresh()
                self.assertEqual(stream.getvalue(), snapshot)
                progress.update(5)
            self.assertGreaterEqual(flush.call_count, 3)
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 3)
        self.assertIn("10/10", lines[-1])
        self.assertTrue("\r" not in stream.getvalue() and "\x1b" not in stream.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")

    def test_nonterminal_refresh_reports_stalled_requests(self) -> None:
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream), mock.patch("aethereval.progress.monotonic") as clock:
            clock.return_value = 0.0
            with Progress(10, "RM scoring") as progress:
                clock.return_value = 10.0
                progress.refresh()
                self.assertEqual(stream.getvalue().count("0/10"), 2)

    def test_disabled_progress_is_silent(self) -> None:
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            with Progress(2, "hidden", enabled=False) as progress:
                progress.update()
                progress.refresh()
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(self.stdout.getvalue(), "")

    def test_failed_preprocessing_does_not_report_completion(self) -> None:
        stream = io.StringIO()
        with mock.patch("sys.stderr", stream):
            with self.assertRaisesRegex(ValueError, "bad prompt"):
                with Progress(10, "preparing") as progress:
                    progress.update(2)
                    raise ValueError("bad prompt")
        self.assertTrue("2/10" in stream.getvalue() and "100%" not in stream.getvalue())
        self.assertEqual(self.stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
