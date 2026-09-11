import pytest

from autoresearch.cli import main


def test_no_command_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: autoresearch" in capsys.readouterr().out
