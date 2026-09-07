import io
from unittest import mock

import pytest

from aethereval.progress import Progress


def test_terminal_refreshes_the_same_line(capsys):
    stream = io.StringIO()
    with (
        mock.patch("sys.stderr", stream),
        mock.patch.object(stream, "isatty", return_value=True),
    ):
        with Progress(10, "preparing", "prompt") as progress:
            progress.update(5)
            progress.refresh()
            assert "50%" in stream.getvalue() and "5/10" in stream.getvalue()
            assert "\n" not in stream.getvalue()
            progress.update(5)
    output = stream.getvalue()
    assert "preparing" in output and "100%" in output and "10/10" in output
    assert "\r" in output and "[aethereval]" not in output
    assert capsys.readouterr().out == ""


def test_nonterminal_emits_flushed_bar_snapshots_without_carriage_returns(capsys):
    stream = io.StringIO()
    with (
        mock.patch("sys.stderr", stream),
        mock.patch("aethereval.progress.monotonic") as clock,
        mock.patch.object(stream, "flush", wraps=stream.flush) as flush,
    ):
        clock.return_value = 0.0
        with Progress(10, "sglang generating") as progress:
            initial = stream.getvalue()
            assert "0/10" in initial and initial.endswith("\n")
            progress.update(5)
            clock.return_value = 9.0
            progress.refresh()
            assert stream.getvalue() == initial
            clock.return_value = 10.0
            progress.refresh()
            assert "5/10" in stream.getvalue()
            snapshot = stream.getvalue()
            clock.return_value = 11.0
            progress.refresh()
            assert stream.getvalue() == snapshot
            progress.update(5)
        assert flush.call_count >= 3
    lines = [line for line in stream.getvalue().splitlines() if line]
    assert len(lines) == 3
    assert "10/10" in lines[-1]
    assert "\r" not in stream.getvalue() and "\x1b" not in stream.getvalue()
    assert capsys.readouterr().out == ""


def test_nonterminal_refresh_reports_stalled_requests():
    stream = io.StringIO()
    with mock.patch("sys.stderr", stream), mock.patch("aethereval.progress.monotonic") as clock:
        clock.return_value = 0.0
        with Progress(10, "RM scoring") as progress:
            clock.return_value = 10.0
            progress.refresh()
            assert stream.getvalue().count("0/10") == 2


def test_disabled_progress_is_silent(capsys):
    with Progress(2, "hidden", enabled=False) as progress:
        progress.update()
        progress.refresh()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_failed_preprocessing_does_not_report_completion(capsys):
    with pytest.raises(ValueError, match="bad prompt"):
        with Progress(10, "preparing") as progress:
            progress.update(2)
            raise ValueError("bad prompt")
    captured = capsys.readouterr()
    assert "2/10" in captured.err and "100%" not in captured.err
    assert captured.out == ""
