import io
from unittest import mock

import pytest

from aethereval.progress import Progress


@pytest.mark.parametrize("is_terminal", [False, True])
def test_progress_uses_only_tqdm_on_stderr(is_terminal, capsys):
    stream = io.StringIO()
    with (
        mock.patch("sys.stderr", stream),
        mock.patch.object(stream, "isatty", return_value=is_terminal),
    ):
        with Progress(10, "preparing", "prompt") as progress:
            progress.update(5)
            progress.refresh()
            assert "50%" in stream.getvalue() and "5/10" in stream.getvalue()
            progress.update(5)
    output = stream.getvalue()
    assert "preparing" in output and "100%" in output and "10/10" in output
    assert "\r" in output and "[aethereval]" not in output
    assert capsys.readouterr().out == ""


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
