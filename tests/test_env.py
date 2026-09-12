"""The Sail API key loader: a real environment variable always beats the file."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch import env


def write_env(tmp_path: Path, text: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(text)
    return p


def test_reads_the_key_from_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(env.KEY, raising=False)
    p = write_env(tmp_path, "# a comment\n\nSAIL_API_KEY='sk_from_file'\nOTHER=2\n")
    assert env.api_key(p) == "sk_from_file"


def test_the_environment_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env.KEY, "sk_from_env")
    p = write_env(tmp_path, "SAIL_API_KEY=sk_from_file\n")
    assert env.api_key(p) == "sk_from_env"


def test_missing_key_exits_with_the_path_in_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(env.KEY, raising=False)
    p = write_env(tmp_path, "OTHER=2\n")
    with pytest.raises(SystemExit) as e:
        env.api_key(p)
    assert str(p.resolve()) in str(e.value)


def test_no_file_is_not_an_error_for_load(tmp_path: Path) -> None:
    env.load_dotenv(tmp_path / "absent")
