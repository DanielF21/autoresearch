"""The target's source at the run's commit, for the Scribe's read tools.

A directory given by the caller is used only when its HEAD is the run's sha, since
the Scribe explains the base code every patch was measured against. Otherwise a
checkout of that one commit is made and cached. Git runs with every ``GIT_``
variable removed, so a hook's ``GIT_INDEX_FILE`` cannot reach it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from autoresearch.intake import scope

GIT_TIMEOUT_S = 600


class SourceError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
        env=scope.git_env(),
    )


def head(repo: Path) -> str | None:
    """HEAD of the repository whose top level is ``repo``, or None."""
    if not repo.is_dir():
        return None
    top = _git(["rev-parse", "--show-toplevel"], repo)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != repo.resolve():
        return None
    r = _git(["rev-parse", "HEAD"], repo)
    return r.stdout.strip() if r.returncode == 0 else None


def checkout_at(url: str, sha: str, dest: Path) -> Path:
    """``dest`` holding ``url`` at ``sha``, reused when it already does."""
    if head(dest) == sha:
        return dest
    if dest.exists():
        raise SourceError(f"{dest} exists and is not {url} at {sha[:12]}; remove it")
    if url.startswith("-"):
        raise SourceError(f"not a repository URL: {url!r}")
    partial = dest.with_name(dest.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for args in (["init", "-q"], ["remote", "add", "origin", url]):
        r = _git(args, partial)
        if r.returncode != 0:
            raise SourceError(f"git {args[0]} failed: {r.stderr.strip()[-300:]}")
    # One commit when the server allows fetching by sha, as GitHub does; the whole
    # history when it does not.
    if _git(["fetch", "-q", "--depth", "1", "origin", sha], partial).returncode != 0:
        r = _git(["fetch", "-q", "origin"], partial)
        if r.returncode != 0:
            raise SourceError(f"git fetch {url} failed: {r.stderr.strip()[-300:]}")
    r = _git(["checkout", "-q", sha], partial)
    if r.returncode != 0 or head(partial) != sha:
        raise SourceError(f"{sha[:12]} is not in {url}: {r.stderr.strip()[-300:]}")
    partial.rename(dest)
    return dest


def source_dir(url: str, sha: str, given: Path | None, cache_root: Path) -> Path:
    if given is not None:
        at = head(given)
        if at != sha:
            raise SourceError(
                f"{given} is at {at[:12] if at else 'no commit'}, the run is at {sha[:12]}"
            )
        return given
    return checkout_at(url, sha, cache_root / f"{scope.repo_name(url)}-{sha[:12]}")
