"""Exercises the incumbent's git operations against a local origin repo. No network."""

import subprocess
from pathlib import Path

import pytest

from autoresearch.incumbent import (
    GitError,
    applies_cleanly,
    apply_and_commit,
    clone_at,
    cumulative_diff,
    head_sha,
    log_shas,
    tree_hash,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "hot.py").write_text("def f(x):\n    return x + 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pinned")
    return repo, _git(repo, "rev-parse", "HEAD")


PATCH = """\
diff --git a/hot.py b/hot.py
--- a/hot.py
+++ b/hot.py
@@ -1,2 +1,2 @@
 def f(x):
-    return x + 1
+    return 1 + x
"""


def test_clone_at_checks_out_the_sha(origin: tuple[Path, str], tmp_path: Path) -> None:
    repo, sha = origin
    dest = tmp_path / "inc"
    assert clone_at(str(repo), sha, dest) == sha
    assert head_sha(dest) == sha


def test_apply_and_commit_advances_head_and_tree(origin: tuple[Path, str], tmp_path: Path) -> None:
    repo, sha = origin
    dest = tmp_path / "inc"
    clone_at(str(repo), sha, dest)
    before_tree = tree_hash(dest)
    assert applies_cleanly(dest, PATCH)
    new = apply_and_commit(dest, PATCH, "accept attempt 0001")
    assert new != sha
    assert head_sha(dest) == new
    assert tree_hash(dest) != before_tree
    assert (dest / "hot.py").read_text() == "def f(x):\n    return 1 + x\n"
    assert log_shas(dest, sha) == (new,)
    assert "+    return 1 + x" in cumulative_diff(dest, sha)


def test_failing_patch_leaves_tree_unchanged(origin: tuple[Path, str], tmp_path: Path) -> None:
    repo, sha = origin
    dest = tmp_path / "inc"
    clone_at(str(repo), sha, dest)
    bad = PATCH.replace("-    return x + 1", "-    return x + 2")
    assert not applies_cleanly(dest, bad)
    with pytest.raises(GitError):
        apply_and_commit(dest, bad, "should fail")
    assert head_sha(dest) == sha
    assert _git(dest, "status", "--porcelain") == ""
