"""The incumbent: a local git repository holding the best code so far.

It starts at the target's pinned commit and gains one commit per accepted
patch. Every function shells out to git and raises GitError with the command
and its stderr on failure, so nothing is silently half applied.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


def clone_at(repo_url: str, sha: str, dest: Path, depth: int = 50) -> str:
    """Clone and check out ``sha``. Returns the full sha. Deepens if the shallow clone missed it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "-q", f"--depth={depth}", repo_url, str(dest)],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        _git(dest, "checkout", "-q", sha)
    except GitError:
        _git(dest, "fetch", "-q", "--unshallow")
        _git(dest, "checkout", "-q", sha)
    _git(dest, "config", "user.email", "referee@autoresearch")
    _git(dest, "config", "user.name", "autoresearch")
    return head_sha(dest)


def head_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def tree_hash(repo: Path, ref: str = "HEAD") -> str:
    """Hash of the tree, independent of commit messages and dates. Used to check a
    box's checkout matches the incumbent without comparing commit ids."""
    return _git(repo, "rev-parse", f"{ref}^{{tree}}").strip()


def applies_cleanly(repo: Path, patch: str) -> bool:
    proc = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo,
        input=patch,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def apply_and_commit(repo: Path, patch: str, message: str) -> str:
    """Apply ``patch`` to the working tree and commit it. Returns the new head sha.

    Raises GitError if the patch does not apply. The working tree is left clean
    either way because ``git apply`` on a failing patch changes nothing.
    """
    _git(repo, "apply", "--index", "-", input_text=patch)
    _git(repo, "commit", "-q", "-m", message)
    return head_sha(repo)


def cumulative_diff(repo: Path, base_sha: str) -> str:
    """Every accepted patch since ``base_sha`` as one diff, for pushing to boxes."""
    return _git(repo, "diff", f"{base_sha}..HEAD")


def log_shas(repo: Path, base_sha: str) -> tuple[str, ...]:
    out = _git(repo, "rev-list", "--reverse", f"{base_sha}..HEAD")
    return tuple(line for line in out.split() if line)
