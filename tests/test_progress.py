from unittest import mock

from aethereval.progress import Progress


def test_server_progress_prints_initial_periodic_and_final_lines(capsys):
    now = [0.0]
    with mock.patch("aethereval.progress.time.monotonic", side_effect=lambda: now[0]):
        with Progress(10, "preparing", "prompt") as progress:
            assert "0/10" in capsys.readouterr().out
            now[0] = 5
            progress.update(2)
            assert capsys.readouterr().out == ""
            now[0] = 10
            progress.update(3)
            line = capsys.readouterr().out
            assert "5/10" in line and "rate=0.50/s" in line and "ETA=10s" in line
            now[0] = 20
            progress.refresh()
            assert "5/10" in capsys.readouterr().out
            progress.update(5)
        output = capsys.readouterr().out
        assert "10/10" in output and "ETA=0s" in output and "\r" not in output
        progress.close()
        assert capsys.readouterr().out == ""


def test_terminal_uses_tqdm_and_disabled_progress_is_silent(capsys):
    with mock.patch("aethereval.progress.sys.stderr.isatty", return_value=True):
        with mock.patch("aethereval.progress.tqdm") as bar:
            with Progress(2, "generating") as progress:
                progress.update()
            bar.return_value.update.assert_called_once_with(1)
            bar.return_value.close.assert_called_once_with()
            assert capsys.readouterr().out == ""
            bar.reset_mock()
            with Progress(2, "hidden", enabled=False) as progress:
                progress.update()
            bar.assert_not_called()
            assert capsys.readouterr().out == ""


def test_failed_preprocessing_does_not_report_completion(capsys):
    try:
        with Progress(10, "preparing") as progress:
            progress.update(2)
            raise ValueError("bad prompt")
    except ValueError:
        pass
    assert "2/10" in capsys.readouterr().out
