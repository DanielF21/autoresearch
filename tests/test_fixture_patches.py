"""The three hand written patches used by Phase 2b check 1. Each must be in scope
and touch only the hot file. Whether they apply to the pinned commit was checked
by hand with git apply on 2026-09-11 and is rechecked by check 1 on a real box."""

from pathlib import Path

import pytest

from autoresearch.config import load_config
from autoresearch.patch import changed_files, changed_line_count, normalised_hash, scope_violations

ROOT = Path(__file__).parent.parent
PATCHES = ROOT / "tests" / "fixtures" / "patches"
NAMES = ("whitespace", "slowdown", "precompute")


@pytest.mark.parametrize("name", NAMES)
def test_fixture_patch_is_in_scope_and_touches_only_the_hot_file(name: str) -> None:
    target = load_config(ROOT / "configs" / "t1_w1.toml").target
    diff = (PATCHES / f"{name}.diff").read_text()
    assert changed_files(diff) == (target.hot_file,)
    assert scope_violations(changed_files(diff), target.allow, target.deny) == ()
    assert changed_line_count(diff) >= 1


def test_fixture_patches_are_distinct() -> None:
    hashes = {normalised_hash((PATCHES / f"{n}.diff").read_text()) for n in NAMES}
    assert len(hashes) == 3
