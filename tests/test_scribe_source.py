"""The Scribe's source checkout: a given directory at the run's sha, or a cached one."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autoresearch.intake import scope
from autoresearch.scribe import source


def _git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scope.git_env(),
    )
    return r.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository with two commits; returns it and both shas, oldest first."""
    src = tmp_path / "origin"
    (src / "pkg").mkdir(parents=True)
    _git(["init", "-q"], src)
    shas = []
    for body in ("LIMIT = 64\n", "LIMIT = 128\n"):
        (src / "pkg" / "mod.py").write_text(body)
        _git(["add", "."], src)
        _git(["commit", "-qm", body.strip()], src)
        shas.append(_git(["rev-parse", "HEAD"], src))
    return src, shas[0], shas[1]


def test_a_given_directory_is_used_only_at_the_runs_sha(tmp_path: Path) -> None:
    src, old, new = _repo(tmp_path)
    assert source.source_dir("file://x", new, src, tmp_path / "cache") == src
    with pytest.raises(source.SourceError, match=f"the run is at {old[:12]}"):
        source.source_dir("file://x", old, src, tmp_path / "cache")
    with pytest.raises(source.SourceError, match="no commit"):
        source.source_dir("file://x", old, src / "pkg", tmp_path / "cache")


def test_a_checkout_of_the_runs_commit_is_made_once_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, old, new = _repo(tmp_path)
    # A pre commit hook sets this; it must not reach the checkout's git.
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "hook-index"))
    url = f"file://{src}"
    dest = source.source_dir(url, old, None, tmp_path / "cache")
    assert dest == tmp_path / "cache" / f"origin-{old[:12]}"
    assert source.head(dest) == old
    assert (dest / "pkg" / "mod.py").read_text() == "LIMIT = 64\n"
    assert source.checkout_at(url, old, dest) == dest
    with pytest.raises(source.SourceError, match="remove it"):
        source.checkout_at(url, new, dest)


def test_a_commit_the_repository_lacks_is_refused(tmp_path: Path) -> None:
    src, _, _ = _repo(tmp_path)
    with pytest.raises(source.SourceError):
        source.checkout_at(f"file://{src}", "b" * 40, tmp_path / "cache" / "x")
    assert not (tmp_path / "cache" / "x").exists()
