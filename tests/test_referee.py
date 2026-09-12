"""The referee against a scripted FakeBox that answers each guest program.

Every test asserts facts in the measurement. There is no verdict to assert.
"""

from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, ok
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig, load_config
from autoresearch.referee.referee import BASE_TREE, PATCHED_TREE, Referee, guest_command
from tests.helpers import BASE_SHA
from tests.helpers import referee_box as make_box

ROOT = Path(__file__).parent.parent
PATCHES = ROOT / "tests" / "fixtures" / "patches"
PRECOMPUTE = (PATCHES / "precompute.diff").read_text()


@pytest.fixture
def config() -> RunConfig:
    return load_config(ROOT / "configs" / "t1_w1.toml")


def _referee(box: FakeBox, config: RunConfig) -> Referee:
    ref = Referee(box, config)
    ref.setup()
    return ref


def test_setup_checks_out_the_base_and_refuses_anything_else(config: RunConfig) -> None:
    box = make_box()
    _referee(box, config)
    assert any(f"git checkout -q --detach {config.target.sha}" in c for c in box.commands)
    with pytest.raises(BoxError, match="not the base"):
        _referee(make_box(head="deadbeef"), config)


def test_real_speedup_is_measured_in_full(config: RunConfig) -> None:
    box = make_box(speedup=1.05)
    ref = _referee(box, config)
    m = ref.measure(PRECOMPUTE)
    assert m.applied and m.apply_error == ""
    assert m.scope_violations == ()
    assert [t.scope for t in m.tests] == ["module", "full"] and m.tests_pass
    assert m.result_matches is True
    assert m.canary_s == 0.5
    assert len(m.pairs) == 6
    assert [p.order for p in m.pairs] == ["base_first", "patched_first"] * 3
    assert [p.hash_seed for p in m.pairs] == [0, 1, 2, 3, 4, 0]
    assert m.speedup == pytest.approx(1.05)
    assert m.noise_floor == 1.0106
    assert m.clears_noise
    assert m.ir is not None and m.ir.base == 1000 and m.ir.patched == 800
    assert m.errors == ()
    assert m.provenance.box_id == "sb_fake"
    assert ref.broken == ""


def test_within_noise_is_recorded_and_not_a_real_speedup(config: RunConfig) -> None:
    m = _referee(make_box(speedup=1.005), config).measure((PATCHES / "whitespace.diff").read_text())
    assert m.tests_pass and m.speedup == pytest.approx(1.005)
    assert not m.clears_noise
    assert m.ir is not None


def test_slowdown_is_measured_with_ratio_below_one(config: RunConfig) -> None:
    m = _referee(make_box(speedup=0.7), config).measure((PATCHES / "slowdown.diff").read_text())
    assert m.speedup == pytest.approx(0.7) and not m.clears_noise


def test_out_of_scope_is_recorded_and_still_measured(config: RunConfig) -> None:
    box = make_box(speedup=1.5)
    diff = (
        "diff --git a/networkx/algorithms/tests/test_cluster.py "
        "b/networkx/algorithms/tests/test_cluster.py\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    )
    m = _referee(box, config).measure(diff)
    assert m.scope_violations and "denied" in m.scope_violations[0]
    assert m.tests_pass and m.speedup == pytest.approx(1.5)
    assert not m.clears_noise  # out of scope can never count as a speedup


def test_patch_that_does_not_apply_records_that_and_nothing_else(config: RunConfig) -> None:
    box = make_box(apply_ok=False)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert not m.applied and "does not apply" in m.apply_error
    assert m.tests == () and m.pairs == () and m.ir is None
    assert not any("run_tests.py" in c or "--repeats" in c for c in box.commands)


def test_empty_diff(config: RunConfig) -> None:
    m = _referee(make_box(), config).measure("nothing here\n")
    assert not m.applied and m.apply_error == "diff touches no files"


def test_test_failures_are_recorded_and_timing_still_runs(config: RunConfig) -> None:
    box = make_box(module_ok=False, full_ok=False, speedup=1.2)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert [t.ok for t in m.tests] == [False, False] and not m.tests_pass
    assert m.speedup == pytest.approx(1.2)
    assert m.ir is not None
    assert not m.clears_noise
    assert any("--scope full" in c for c in box.commands)


def test_different_result_is_recorded_and_timing_still_runs(config: RunConfig) -> None:
    m = _referee(make_box(speedup=2.0, fp_match=False), config).measure(PRECOMPUTE)
    assert m.result_matches is False
    assert m.speedup == pytest.approx(2.0)
    assert not m.clears_noise


def test_patched_tree_that_cannot_run_skips_timing_and_counts(config: RunConfig) -> None:
    box = make_box(patched_verify_fails=True)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert m.tests_pass  # the tests ran before the verify step
    assert m.base_fp == "fp_same" and m.patched_fp == ""
    assert m.pairs == () and m.speedup is None and m.ir is None
    assert any("verify patched" in e for e in m.errors)
    assert any("skipped" in e for e in m.errors)
    assert not any("--repeats" in c for c in box.commands)


def test_contaminated_pairs_retry_once_then_record_an_error(config: RunConfig) -> None:
    box = make_box(contaminate_pairs=True)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert m.speedup is None
    assert any("clean pairs" in e for e in m.errors)
    launches = [c for c in box.commands if "time_target.py" in c and "--repeats" in c]
    assert len(launches) == 2 * 6 * 2  # two passes of six pairs, two launches each
    assert all(p.contaminated for p in m.pairs) and m.pairs[0].reasons == ("steal",)
    assert m.ir is not None  # instruction counts still ran


def test_ir_failure_is_an_error_not_a_lost_measurement(config: RunConfig) -> None:
    m = _referee(make_box(speedup=1.05, ir_ok=False), config).measure(PRECOMPUTE)
    assert m.ir is None and any("instruction counts" in e for e in m.errors)
    assert m.clears_noise


def test_timing_launches_are_pinned_seeded_and_alternate(config: RunConfig) -> None:
    box = make_box()
    _referee(box, config).measure(PRECOMPUTE)
    launches = [c for c in box.commands if "time_target.py" in c and "--repeats" in c]
    assert launches and all("taskset -c 2" in c for c in launches)
    assert BASE_TREE in launches[0] and PATCHED_TREE in launches[1]
    assert PATCHED_TREE in launches[2] and BASE_TREE in launches[3]


def test_cleanup_runs_after_every_measurement_and_marks_broken_on_failure(
    config: RunConfig,
) -> None:
    box = make_box(cleanup_ok=False, apply_ok=False)
    ref = _referee(box, config)
    ref.measure(PRECOMPUTE)
    assert len([c for c in box.commands if "--remove" in c]) == 2
    assert "cleanup" in ref.broken


def test_box_error_propagates_to_the_orchestrator(config: RunConfig) -> None:
    box = make_box()
    ref = _referee(box, config)
    box.terminate()
    with pytest.raises(BoxError):
        ref.measure(PRECOMPUTE)


def test_worktrees_come_from_the_base_sha(config: RunConfig) -> None:
    box = make_box()
    _referee(box, config).measure(PRECOMPUTE)
    adds = [c for c in box.commands if "apply_patch.py" in c and "--commit" in c]
    assert len(adds) == 2 and all(f"--commit {BASE_SHA}" in c for c in adds)


def test_guest_command_quotes_and_pins() -> None:
    cmd = guest_command("time_target.py", "--graph", "nx.g(1, 0.5)", pin=2)
    assert (
        cmd == "cd /workspace/guest && taskset -c 2 python3 time_target.py --graph 'nx.g(1, 0.5)'"
    )


def test_fake_box_ok_helper_is_used() -> None:
    assert ok("x").stdout == "x"
