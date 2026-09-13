"""Reading a run, the hard filters, and the code computed facts about a diff."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.scribe import facts as facts_mod
from autoresearch.scribe import filters, runread
from autoresearch.scribe.config import ScribeConfigError, load_scribe_config, parse_scribe_config
from tests.scribe_helpers import FIXTURES, make_run

ROOT = Path(__file__).parent.parent
PATCH = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"


def test_candidates_flatten_measurements_and_read_both_setup_shapes(tmp_path: Path) -> None:
    run = make_run(
        tmp_path,
        [
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}},
            {"patch": None},
        ],
    )
    target = runread.read_target(run)
    assert target.setups == {"dense": "G = make(100, 0.5)", "sparse": "G = make(100, 0.01)"}
    first, second = runread.load_candidates(run)
    assert first.number == 1 and first.clears_noise and first.tests_pass
    assert first.speedup == pytest.approx((3.0 * 1.5) ** 0.5)
    dense = first.inputs[0]
    assert dense.setup == "G = make(100, 0.5)"
    assert dense.clean_pairs == 6 and dense.base_median_s == 1.0
    assert dense.patched_median_s == pytest.approx(1 / 3)
    assert [t.scope for t in first.tests] == ["module", "full"]
    assert second.patch is None and second.error == "box went away" and not second.measured


def test_attempt_files_never_expose_config_or_transcripts_unless_asked(tmp_path: Path) -> None:
    run = make_run(tmp_path, [{"patch": PATCH, "speedups": {"dense": 2.0, "sparse": 2.0}}])
    closed = runread.attempt_files(run, 1, transcripts=False)
    assert set(closed) == {
        "attempts/0001/patch.diff",
        "attempts/0001/measurement.json",
        "attempts/0001/rationale.md",
    }
    assert "attempts/0001/transcript.jsonl" in runread.attempt_files(run, 1, transcripts=True)
    assert set(runread.profile_files(run)) == {"target/profile_inputs.txt"}


def test_filters_record_every_reason_and_separate_reviewable_from_writable(tmp_path: Path) -> None:
    run = make_run(
        tmp_path,
        [
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 0.5}},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}, "duplicate_of": "0001"},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}, "tests_ok": False},
            {"patch": None},
        ],
    )
    got = {
        c.number: filters.filter_candidate(c, min_inputs=2) for c in runread.load_candidates(run)
    }
    assert got[1].writable and got[1].reasons == ()
    assert not got[2].reviewable and any("regresses on: sparse" in r for r in got[2].reasons)
    assert any("duplicate of attempt 0001" in r for r in got[3].reasons)
    assert any("tests did not pass" in r for r in got[4].reasons)
    assert got[5].reasons == ("no patch (no_patch)",)

    three = filters.filter_candidate(runread.load_candidates(run)[0], min_inputs=3)
    assert three.reviewable and not three.writable
    assert "fewer than 3" in three.reasons[-1]


def test_facts_from_the_diff_alone_find_the_dense_patch_import_and_guard() -> None:
    """t1_w4b 0010 is the 69x dense rewrite that regresses on sparse graphs, and
    0001 the bitset patch that does not (artifacts/generality.md section 1)."""
    dense = facts_mod.compute_facts((FIXTURES / "t1_w4b_0010.diff").read_text())
    assert "numpy" in dense.new_imports
    assert any("<= 1400" in c for c in dense.new_condition_numbers)
    safe = facts_mod.compute_facts((FIXTURES / "t1_w4b_0001.diff").read_text())
    assert safe.new_imports == ()
    assert safe.files == ("networkx/algorithms/cluster.py",)
    assert safe.changed_lines == safe.added + safe.removed > 0


BASE = """\
import math

__all__ = ["area"]


def area(r):
    return math.pi * r * r


def perimeter(r):
    return 2 * math.pi * r
"""

PATCHED = """\
import math
import numpy as np

__all__ = ["area"]


def area(r, fast=False):
    if r.size > 5000:
        return np.pi * r._data ** 2
    return math.pi * r * r


def perimeter(r):
    return 2 * math.pi * r
"""


def test_ast_facts_name_functions_api_changes_guards_and_private_reads(tmp_path: Path) -> None:
    from tests.scribe_helpers import diff_in, make_upstream

    repo, _ = make_upstream(tmp_path, {"geo.py": BASE})
    diff = diff_in(repo, "geo.py", PATCHED)
    f = facts_mod.compute_facts(diff, {"geo.py": BASE}, {"geo.py": PATCHED})
    names = {fc.name for fc in f.functions}
    assert "area" in names and "perimeter" not in names
    assert f.new_imports == ("numpy",)
    assert f.public_api_changes == ("geo.py: signature of area changed",)
    assert any("5000" in c for c in f.new_condition_numbers)
    assert any("r._data" in p for p in f.private_attribute_reads)
    assert "5000" in f.numeric_literals
    area = next(fc for fc in f.functions if fc.name == "area")
    assert area.removed == 1 and not area.old_path_kept


def test_the_repository_config_parses() -> None:
    cfg = load_scribe_config(ROOT / "configs" / "scribe" / "scribe.toml", ROOT)
    assert cfg.corpus.n == 20 and cfg.corpus.holdout_n < cfg.corpus.n
    assert set(cfg.models) == {"reviewer", "writer", "judge"}
    assert cfg.method.is_dir()


def test_config_refuses_a_corpus_size_outside_the_range(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "scribe" / "scribe.toml").read_text().replace("n = 20", "n = 30")
    with pytest.raises(ScribeConfigError, match="between 10 and 20"):
        parse_scribe_config(text, tmp_path / "s.toml", ROOT)
