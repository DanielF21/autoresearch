"""The one sailbox image every box in a run is built from, and the layout inside it.

Built from code so nothing expires: Debian, the tools the referee needs, the
target's own dependencies, and the target repository cloned at its pinned
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

# What every box has whatever the target. util-linux is here for taskset, which
# pins every timing launch to one core. The target's own dependencies come from
# its config; a box holds nothing the target did not ask for.
APT_PACKAGES = ("git", "curl", "build-essential", "time", "util-linux")
PIP_PACKAGES = ("pytest", "pytest-xdist")


def apt_packages(target: TargetSpec) -> tuple[str, ...]:
    return (*APT_PACKAGES, *target.apt)


def pip_packages(target: TargetSpec) -> tuple[str, ...]:
    """The target's packages first, then the harness's own.

    That order is what the image had before a target could name its
    dependencies, when three of them were installed for everyone, so the
    image a legacy networkx config composes is byte for byte the one its runs
    were built on and resumes without a rebuild.
    """
    return (*target.pip, *PIP_PACKAGES)


def clone_commands(target: TargetSpec) -> tuple[str, ...]:
    """Shell steps that put the target at its pinned commit under REPO_DIR.

    Submodules are fetched only if the repository declares any, so a target
    that keeps test data or a vendored dependency in one is not cloned as half
    a tree.
    """
    return (
        f"git clone -q {target.repo} {REPO_DIR}",
        f"cd {REPO_DIR} && git checkout -q {target.sha}",
        f"cd {REPO_DIR} && if [ -f .gitmodules ]; then git submodule update --init --recursive -q; fi",
        f"cd {REPO_DIR} && git config user.email referee@autoresearch && git config user.name autoresearch",
        f"mkdir -p {WORK_DIR}",
    )


def build_image(target: TargetSpec) -> Any:
    """The Sail image definition. Imported lazily so tests never need the SDK."""
    import sail

    image = sail.Image.debian_amd64.apt_install(*apt_packages(target)).pip_install(
        *pip_packages(target)
    )
    return image.run_commands(*clone_commands(target))
