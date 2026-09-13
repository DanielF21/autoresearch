"""Local source of the target: one clone per repo, one worktree per tree.

The Scribe reads code on this machine, never in a box. A tree is the base commit,
or the base commit with one candidate's patch applied, keyed by the patch's hash
so a candidate reviewed twice is checked out once. Written for git 2.15, which has
no ``worktree remove``: a stale tree is deleted and the worktree list pruned.

Every git call goes through ``GitRunner`` as an argument list, never a shell.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from autoresearch.scribe.layout import sha256_text

CLONE_TIMEOUT = 1800.0


class GitError(RuntimeError):
    pass


class PatchApplyError(GitError):
    pass


class GitRunner(Protocol):
    def run(self, args: list[str], cwd: Path, timeout: float = 300.0) -> str: ...


class SubprocessGit:
    def run(self, args: list[str], cwd: Path, timeout: float = 300.0) -> str:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}
        try:
            r = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise GitError(f"git {args[0]} timed out after {timeout:.0f}s") from e
        if r.returncode != 0:
            raise GitError(f"git {args[0]} failed ({r.returncode}): {r.stderr.strip()[-500:]}")
        return r.stdout


def slug(url: str) -> str:
    bare = re.sub(r"^[a-z]+://", "", url).removesuffix(".git").strip("/")
    return re.sub(r"[^A-Za-z0-9._]+", "__", bare)


class RepoCache:
    def __init__(self, root: Path, git: GitRunner | None = None) -> None:
        self.root = root
        self.git: GitRunner = git if git is not None else SubprocessGit()

    def clone(self, url: str, sha: str) -> Path:
        dest = self.root / "clones" / slug(url)
        if not (dest / ".git").exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.git.run(
                ["clone", "--quiet", "--no-checkout", url, str(dest)],
                cwd=dest.parent,
                timeout=CLONE_TIMEOUT,
            )
        if not self._has_commit(dest, sha):
            self.git.run(["fetch", "--quiet", "origin"], cwd=dest, timeout=CLONE_TIMEOUT)
            if not self._has_commit(dest, sha):
                raise GitError(f"{sha} is not in {url}")
        return dest

    def _has_commit(self, clone: Path, sha: str) -> bool:
        try:
            self.git.run(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=clone)
        except GitError:
            return False
        return True

    def tree(self, url: str, sha: str, patch: str | None = None) -> Path:
        """The base tree, or the base with ``patch`` applied. Raises PatchApplyError."""
        clone = self.clone(url, sha)
        key = "base" if patch is None else sha256_text(patch)[:12]
        path = self.root / "trees" / sha[:12] / key
        ready = path.parent / f"{key}.ready"
        if ready.exists() and path.exists():
            return path
        if path.exists():
            shutil.rmtree(path)
        self.git.run(["worktree", "prune"], cwd=clone)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.git.run(["worktree", "add", "--detach", str(path), sha], cwd=clone)
        if patch is not None:
            patch_file = path.parent / f"{key}.diff"
            patch_file.write_text(patch)
            try:
                self.git.run(["apply", "--check", str(patch_file)], cwd=path)
                self.git.run(["apply", str(patch_file)], cwd=path)
            except GitError as e:
                shutil.rmtree(path, ignore_errors=True)
                self.git.run(["worktree", "prune"], cwd=clone)
                raise PatchApplyError(f"patch does not apply to {sha[:12]}: {e}") from e
        ready.write_text(sha + "\n")
        return path


def read_sources(tree: Path, files: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in files:
        p = tree / rel
        if p.is_file():
            out[rel] = p.read_text(errors="replace")
    return out
