"""The one sailbox image every box in a run is built from, and the layout inside it.

Built from code so nothing expires: Debian, the tools the referee needs, the
target's Python dependencies, and the target repository cloned at its pinned
commit under ``REPO_DIR``. The package is never installed; every guest program
imports it by path from a git worktree, so no installed copy can shadow the tree
under test.

The directory constants below are the whole of ``/workspace``. They are here
rather than beside their users because the worker is now told this layout in
words, so there must be one place that defines it.

Sail caches image builds by content, so this is paid once per distinct target.
"""

from __future__ import annotations

from typing import Any

from autoresearch.config import TargetSpec

REPO_DIR = "/workspace/repo"  # the target, checked out at the base commit
GUEST_DIR = "/workspace/guest"  # the measurement programs
WORK_DIR = "/workspace/work"  # the referee's worktrees and logs
BASE_DIR = "/workspace/base"  # worker only: a read only worktree of the base commit
HISTORY_DIR = "/workspace/history"  # worker only: every earlier attempt

APT_PACKAGES = ("git", "curl", "build-essential", "valgrind", "time", "util-linux")
PIP_PACKAGES = ("numpy", "scipy", "pandas", "pytest", "pytest-xdist")


def clone_commands(target: TargetSpec) -> tuple[str, ...]:
    """Shell steps that put the target at its pinned commit under REPO_DIR."""
    return (
        f"git clone -q {target.repo} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout -q {target.sha}",
        f"cd {REPO_DIR} && git config user.email referee@autoresearch && git config user.name autoresearch",
        f"mkdir -p {WORK_DIR}",
    )


def build_image(target: TargetSpec) -> Any:
    """The Sail image definition. Imported lazily so tests never need the SDK."""
    import sail

    image = sail.Image.debian_amd64.apt_install(*APT_PACKAGES).pip_install(*PIP_PACKAGES)
    return image.run_commands(*clone_commands(target))
