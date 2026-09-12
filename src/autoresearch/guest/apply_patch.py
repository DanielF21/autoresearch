"""Create a git worktree at a commit and apply a patch to it. Prints one JSON object.

The referee needs a clean tree per attempt. A worktree is cheap, shares the
object store, and is removed with ``--remove`` when the attempt is done. The
patch is checked before it is applied, so a failing patch leaves the worktree
identical to the commit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _git(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    # The repo is always named explicitly, so inherited GIT_* variables (a git
    # hook sets GIT_DIR and GIT_INDEX_FILE) must not redirect the command.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _remove_worktree(repo: Path, worktree: Path) -> None:
    """Delete the directory, then let git forget it. Works on every git version."""
    shutil.rmtree(worktree, ignore_errors=True)
    _git(repo, "worktree", "prune")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--patch", default="", help="path to a diff; omit for a pristine worktree")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    worktree = Path(args.worktree).resolve()

    if args.remove:
        _remove_worktree(repo, worktree)
        print(json.dumps({"kind": "worktree_removed", "ok": not worktree.exists(), "error": ""}))
        return 0 if not worktree.exists() else 1

    if not args.commit:
        print(json.dumps({"kind": "worktree", "ok": False, "error": "--commit is required"}))
        return 2

    if worktree.exists():
        _remove_worktree(repo, worktree)
    proc = _git(repo, "worktree", "add", "--detach", str(worktree), args.commit)
    if proc.returncode != 0:
        print(json.dumps({"kind": "worktree", "ok": False, "error": proc.stderr.strip()}))
        return 1

    applied = False
    changed: list[str] = []
    if args.patch:
        patch_text = Path(args.patch).read_text()
        check = _git(worktree, "apply", "--check", "-", input_text=patch_text)
        if check.returncode != 0:
            print(json.dumps({"kind": "worktree", "ok": False, "error": check.stderr.strip()}))
            return 1
        apply = _git(worktree, "apply", "-", input_text=patch_text)
        if apply.returncode != 0:
            print(json.dumps({"kind": "worktree", "ok": False, "error": apply.stderr.strip()}))
            return 1
        applied = True
        changed = [line for line in _git(worktree, "diff", "--name-only").stdout.split() if line]

    head = _git(worktree, "rev-parse", "HEAD").stdout.strip()
    print(
        json.dumps(
            {
                "kind": "worktree",
                "ok": True,
                "worktree": str(worktree),
                "head": head,
                "applied": applied,
                "changed_files": changed,
                "error": "",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
