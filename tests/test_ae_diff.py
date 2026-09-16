"""A program's files as a patch: the harness reads it and git applies it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from alphaevolve.blocks import apply_edits, assemble, derive, parse_edits
from alphaevolve.diff import build_patch, unified
from autoresearch.patch import changed_files, scope_violations
from tests.ae_helpers import FLAT, HOT, SOURCE, edit_reply

GUEST = Path(__file__).parent.parent / "src" / "autoresearch" / "guest"
OTHER = "networkx/utils/misc.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def test_unchanged_files_make_no_patch_and_texts_must_end_in_a_newline() -> None:
    assert build_patch({HOT: SOURCE}, {HOT: SOURCE}) is None
    with pytest.raises(ValueError, match="newline"):
        unified(HOT, SOURCE, "no newline")


def test_the_patch_names_its_files_for_the_harness() -> None:
    patch = build_patch(
        {HOT: SOURCE, OTHER: "x = 1\n"},
        {HOT: SOURCE.replace("total = 0", "total = 1"), OTHER: "x = 2\n"},
    )
    assert patch is not None
    assert patch.startswith(f"diff --git a/{HOT} b/{HOT}\n--- a/{HOT}\n+++ b/{HOT}\n")
    assert changed_files(patch) == (HOT, OTHER)
    assert scope_violations(changed_files(patch), ("networkx/**",), ("tests/**",)) == ()


def test_the_guest_applies_an_edited_program(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "networkx" / "algorithms").mkdir(parents=True)
    (repo / HOT).write_text(SOURCE)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    sha = _git(repo, "rev-parse", "HEAD").strip()

    blocks = derive(FLAT, {HOT: SOURCE}, ("networkx/**",), (), 10_000)
    texts = apply_edits(
        [b.text for b in blocks],
        parse_edits(edit_reply("        total += 1", "        total += 2")),
    )
    files = assemble({HOT: SOURCE}, blocks, texts)
    patch = build_patch({HOT: SOURCE}, files)
    assert patch is not None
    patch_file = tmp_path / "attempt.diff"
    patch_file.write_text(patch)
    wt = tmp_path / "wt"
    proc = subprocess.run(
        [
            sys.executable,
            str(GUEST / "apply_patch.py"),
            "--repo",
            str(repo),
            "--worktree",
            str(wt),
            "--commit",
            sha,
            "--patch",
            str(patch_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ok"] is True and out["changed_files"] == [HOT], proc.stderr
    assert (wt / HOT).read_text() == files[HOT]
