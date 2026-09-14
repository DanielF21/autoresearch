"""The referee against a scripted FakeBox that answers each guest program.

Every test asserts facts in the measurement. There is no verdict to assert.
"""

import shlex
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, ok
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig, SuitePaths, load_config
from autoresearch.referee.referee import BASE_TREE, PATCHED_TREE, Referee, guest_command
from autoresearch.types import Attempt, AttemptRef, StopReason, Usage
from tests.helpers import BASE_SHA, arg
from tests.helpers import referee_box as make_box

ROOT = Path(__file__).parent.parent
PATCHES = ROOT / "tests" / "fixtures" / "patches"
PRECOMPUTE = (PATCHES / "precompute.diff").read_text()


# Every input is timed with its own pairs, so any count of launches or pairs is
# per input. Read from the config so adding an input does not silently pass.
def _n_inputs(config: RunConfig) -> int:
    return len(config.target.inputs)


def _first(m, name: str = ""):  # type: ignore[no-untyped-def]
    """One input's timing, the first unless named."""
    return next(i for i in m.inputs if not name or i.name == name)


@pytest.fixture
def config() -> RunConfig:
    return load_config(ROOT / "configs" / "t1_w4d.toml")


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
    assert [i.name for i in m.inputs] == [i.name for i in config.target.inputs]
    for timed in m.inputs:
        assert len(timed.pairs) == 6
        assert [p.order for p in timed.pairs] == ["base_first", "patched_first"] * 3
        assert [p.hash_seed for p in timed.pairs] == [0, 1, 2, 3, 4, 0]
        assert timed.speedup == pytest.approx(1.05)
        assert not timed.retried and timed.errors == ()
    assert _first(m).noise_floor == 1.0106
    assert m.speedup == pytest.approx(1.05)  # geomean of the same ratio everywhere
    assert m.worst_speedup == pytest.approx(1.05)
    assert m.regressions == ()
    assert m.clears_noise
    assert m.errors == ()
    assert m.provenance.box_id == "sb_fake"
    assert ref.broken == ""


def test_within_noise_is_recorded_and_not_a_real_speedup(config: RunConfig) -> None:
    m = _referee(make_box(speedup=1.005), config).measure((PATCHES / "whitespace.diff").read_text())
    assert m.tests_pass and m.speedup == pytest.approx(1.005)
    assert not m.clears_noise


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
    assert m.tests == ()
    # The inputs are still listed, all untimed: the record says what would have
    # been measured as well as that nothing was.
    assert all(i.pairs == () and i.speedup is None for i in m.inputs)
    assert not any("run_tests.py" in c or "--repeats" in c for c in box.commands)


def test_empty_diff(config: RunConfig) -> None:
    m = _referee(make_box(), config).measure("nothing here\n")
    assert not m.applied and m.apply_error == "diff touches no files"


def test_test_failures_are_recorded_and_timing_still_runs(config: RunConfig) -> None:
    box = make_box(module_ok=False, full_ok=False, speedup=1.2)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert [t.ok for t in m.tests] == [False, False] and not m.tests_pass
    assert m.speedup == pytest.approx(1.2)
    assert not m.clears_noise
    assert any("--scope full" in c for c in box.commands)


def test_different_result_is_recorded_and_timing_still_runs(config: RunConfig) -> None:
    m = _referee(make_box(speedup=2.0, fp_match=False), config).measure(PRECOMPUTE)
    assert m.result_matches is False
    assert m.speedup == pytest.approx(2.0)
    assert not m.clears_noise


def test_patched_tree_that_cannot_run_skips_timing(config: RunConfig) -> None:
    box = make_box(patched_verify_fails=True)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert m.tests_pass  # the tests ran before the verify step
    assert _first(m).base_fp == "fp_same_0" and _first(m).patched_fp == ""
    assert all(i.pairs == () and i.speedup is None for i in m.inputs)
    assert m.speedup is None
    assert any("verify patched" in e for e in m.errors)
    # The skip is recorded on each input, since it is per input now.
    assert all(any("skipped" in e for e in i.errors) for i in m.inputs)
    assert not any("--repeats" in c for c in box.commands)


def test_contaminated_pairs_retry_then_leave_the_input_untimed(config: RunConfig) -> None:
    box = make_box(contaminate_pairs=True)
    m = _referee(box, config).measure(PRECOMPUTE)
    n = _n_inputs(config)
    retries = config.referee.timing_retries
    assert retries == 2  # the default; the frozen config does not set it
    assert m.speedup is None
    assert all(i.retried and i.retries == retries for i in m.inputs)
    assert all(any("untimed" in e for e in i.errors) for i in m.inputs)
    assert m.untimed == tuple(i.name for i in config.target.inputs)
    assert not m.clears_noise
    launches = [c for c in box.commands if "time_target.py" in c and "--repeats" in c]
    # One pass plus every retry, six pairs of two launches each, for every input.
    assert len(launches) == n * (1 + retries) * 6 * 2
    assert all(p.contaminated for p in _first(m).pairs)
    assert _first(m).pairs[0].reasons == ("steal",)


def test_a_retry_that_comes_back_clean_times_the_input(config: RunConfig) -> None:
    # The first pass of every input is tainted, twelve launches; the retry is clean.
    box = make_box(speedup=1.5, contaminate_pairs=12)
    m = _referee(box, config).measure(PRECOMPUTE)
    assert all(i.retried and i.retries == 1 and i.errors == () for i in m.inputs)
    assert all(i.speedup == pytest.approx(1.5) for i in m.inputs)
    assert m.untimed == () and m.clears_noise


def test_an_attempt_with_an_untimed_input_is_named_and_never_a_real_speedup(
    config: RunConfig,
) -> None:
    """The mean over the inputs that were timed is kept, but it cannot be a record.

    The width experiment's two largest recorded speedups were attempts whose
    cheap sparse inputs never timed; the mean over the rest was 34x and 32x
    against 23x and 27x with all five.
    """
    from autoresearch.worker.prompt import outcome

    first = config.target.inputs[0].name
    box = make_box(speedup=1.5)
    m = _referee(box, config).measure(PRECOMPUTE)
    inputs = tuple(replace(i, speedup=None, pairs=()) if i.name == first else i for i in m.inputs)
    partial = replace(m, inputs=inputs)
    assert partial.speedup == pytest.approx(1.5) and partial.result_matches is True
    assert partial.untimed == (first,) and not partial.clears_noise
    attempt = Attempt(
        ref=AttemptRef(1, 1, 0),
        base_sha=config.target.sha,
        patch=PRECOMPUTE,
        prediction=None,
        rationale="",
        stop_reason=StopReason.SUBMITTED,
        usage=Usage(),
        wall_s=1.0,
        measurement=partial,
    )
    assert outcome(attempt) == f"timing failed on {first}"


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


def test_every_input_is_timed_with_its_own_setup(config: RunConfig) -> None:
    box = make_box()
    _referee(box, config).measure(PRECOMPUTE)
    launches = [c for c in box.commands if "time_target.py" in c and "--repeats" in c]
    # Six pairs of two launches on each of two instances: the held out one, then the shown.
    assert len(launches) == _n_inputs(config) * 6 * 2 * 2
    for spec in config.target.inputs:
        mine = [c for c in launches if f"--label {spec.name}/" in c]
        assert len(mine) == 24, f"{spec.name} should get six pairs of two launches twice"
        assert all(f"--setup {shlex.quote(spec.setup)}" in c for c in mine)
        assert all(f"--call {shlex.quote(config.target.call)}" in c for c in mine)
        seeds = [arg(c, "--seed") for c in mine]
        assert seeds[12:] == ["0"] * 12 and len(set(seeds[:12])) == 1 and seeds[0] != "0"


def test_every_launch_carries_the_whole_target_and_nothing_is_defaulted(
    config: RunConfig,
) -> None:
    """The guest has no idea what it is timing; every fact travels as a flag."""
    box = make_box()
    _referee(box, config).measure(PRECOMPUTE)
    t = config.target
    for c in (c for c in box.commands if "time_target.py" in c):
        assert f"--package {t.package}" in c and f"--alias {t.alias}" in c
        assert f"--package-root {t.package_root}" in c and f"--hot {t.hot_file}" in c
        assert "--graph" not in c
        # The whole result is hashed; there is no expression a target could
        # name to narrow what is compared.
        assert "--fingerprint" not in c


def test_the_full_suite_is_the_configured_path_not_derived_from_the_hot_file(
    config: RunConfig,
) -> None:
    other = replace(
        config, target=replace(config.target, tests=SuitePaths(module="t/test_a.py", full="t"))
    )
    box = make_box()
    _referee(box, other).measure(PRECOMPUTE)
    runs = [c for c in box.commands if "run_tests.py" in c]
    assert any("--target t/test_a.py" in c and "--scope module" in c for c in runs)
    assert any("--target t " in c and "--scope full" in c for c in runs)
    assert all(f"--package-root {config.target.package_root}" in c for c in runs)
    assert not any("--target networkx " in c for c in runs)


def test_measure_refuses_an_uncalibrated_input_by_name(config: RunConfig) -> None:
    """calibrate and check accept a floorless input; a measurement cannot."""
    first, *rest = config.target.inputs
    bare = replace(
        config, target=replace(config.target, inputs=(replace(first, noise_floor=None), *rest))
    )
    ref = _referee(make_box(), bare)
    with pytest.raises(ValueError, match=first.name):
        ref.measure(PRECOMPUTE)
    # null_pairs never needs a floor, so calibration still runs.
    assert set(ref.null_pairs(rounds=1)) == {i.name for i in config.target.inputs}


def test_each_pair_records_what_both_launches_paid_before_their_first_call(
    config: RunConfig,
) -> None:
    m = _referee(make_box(), config).measure(PRECOMPUTE)
    for p in _first(m).pairs:
        assert p.base_fixed_s == 0.29 and p.patched_fixed_s == 0.31


def test_a_patch_fast_on_one_input_and_slower_on_another_is_not_a_speedup(
    config: RunConfig,
) -> None:
    """The defect the input set exists to catch. artifacts/generality.md.

    Patch 0010's shape, faked: huge on the dense input, less than half speed on
    the scale free one. The geometric mean is well above any floor and the
    measurement must still refuse to call it a real speedup.
    """
    box = make_box(speedup=20.0, per_input={"gn800": 0.22})
    m = _referee(box, config).measure(PRECOMPUTE)
    assert m.tests_pass and m.result_matches is True
    assert m.speedup is not None and m.speedup > 5.0
    assert m.worst_speedup == pytest.approx(0.22)
    assert m.regressions == ("gn800",)
    assert not m.clears_noise


def test_one_input_that_cannot_run_does_not_stop_the_others(config: RunConfig) -> None:
    """A per input skip, so one broken input costs its own launches and no more."""
    gn = next(i for i in config.target.inputs if i.name == "gn800")
    box = make_box(verify_fails_for=(gn.setup,))
    m = _referee(box, config).measure(PRECOMPUTE)

    assert [i.name for i in m.inputs if i.speedup is None] == ["gn800"]
    assert len([i for i in m.inputs if i.speedup is not None]) == _n_inputs(config) - 1
    assert m.speedup is not None  # the rest still produced a mean
    skipped = next(i for i in m.inputs if i.name == "gn800")
    assert any("skipped" in e for e in skipped.errors)
    # It cost two verify launches and no timing launches.
    timing = [c for c in box.commands if "--repeats" in c and gn.setup in c]
    assert timing == []


def test_null_pairs_times_two_unpatched_trees_with_the_referees_own_plan(
    config: RunConfig,
) -> None:
    """Calibration has to measure the statistic it calibrates.

    So the pairs come back through the same ``_time_pairs`` the measurement uses,
    both trees are made without a patch, and no test suite or canary runs.
    """
    box = make_box()
    ref = _referee(box, config)
    measured = ref.null_pairs(rounds=2)

    assert set(measured) == {i.name for i in config.target.inputs}
    for pairs in measured.values():
        assert len(pairs) == 2 * config.referee.pairs
    assert not any("run_tests.py" in c for c in box.commands)
    assert not any("canary.py" in c for c in box.commands)
    assert not any("--patch" in c for c in box.commands)
    # Both worktrees made, both removed.
    assert sum(1 for c in box.commands if "--worktree" in c and "--remove" not in c) == 2
    assert sum(1 for c in box.commands if "--remove" in c) == 2


def test_null_pairs_can_be_limited_to_named_inputs(config: RunConfig) -> None:
    ref = _referee(make_box(), config)
    measured = ref.null_pairs(rounds=1, only=("gn800",))
    assert set(measured) == {"gn800"}
    with pytest.raises(ValueError, match="no such input: nope"):
        _referee(make_box(), config).null_pairs(rounds=1, only=("nope",))


def test_guest_command_quotes_and_pins() -> None:
    cmd = guest_command("time_target.py", "--setup", "G = nx.g(1, 0.5)", pin=2)
    assert (
        cmd
        == "cd /workspace/guest && taskset -c 2 python3 time_target.py --setup 'G = nx.g(1, 0.5)'"
    )


def test_fake_box_ok_helper_is_used() -> None:
    assert ok("x").stdout == "x"


def test_a_cached_result_on_repeated_calls_is_memoized_not_a_speedup(config: RunConfig) -> None:
    """t1_p2_w16 from round 5: the whole answer stored in the graph's cache dict,
    every call after the first free, 500x to 6400x recorded as real. The guest now
    reports each launch's first call apart from the rest, and the gap on the
    patched tree alone is the tell."""
    box = make_box(speedup=2000.0, memoized_inputs=("er1000_005", "gn800"))
    m = _referee(box, config).measure(PRECOMPUTE)
    assert m.tests_pass and m.result_matches is True
    assert m.speedup is not None and m.speedup > 100
    assert m.memoized == ("er1000_005", "gn800")
    assert _first(m, "er1000_005").memoized and not _first(m, "er1000_001").memoized
    assert any("memoized" in e for e in _first(m, "gn800").errors)
    assert not m.clears_noise
    p = _first(m, "er1000_005").pairs[0]
    assert p.patched_first_s == 1.0 and p.patched_warm_s == pytest.approx(1 / 2000.0)
    assert p.base_first_s == 1.0 and p.base_warm_s == 1.0


def test_an_honest_speedup_is_not_memoized(config: RunConfig) -> None:
    m = _referee(make_box(speedup=30.0), config).measure(PRECOMPUTE)
    assert m.clears_noise and m.memoized == () and m.overfit == ()
    p = _first(m).pairs[0]
    assert p.patched_first_s == p.patched_warm_s
    # The score is the held out instance's; the shown one is recorded beside it.
    i = _first(m)
    assert i.seed != 0 and all(p.seed == i.seed for p in i.pairs)
    assert all(p.seed == 0 for p in i.shown_pairs) and len(i.shown_pairs) == 6
    assert i.speedup == pytest.approx(30.0) and i.shown_speedup == pytest.approx(30.0)
    assert i.held_out_base_fp == i.held_out_patched_fp == f"fp_same_{i.seed}"


def test_a_patch_that_recognises_the_shown_instance_is_overfit_not_a_speedup(
    config: RunConfig,
) -> None:
    """pycodestyle_p3_w16 attempts 0233 and 0247: the six benchmark files
    regenerated at import and matched by string equality, 2985x and 3470x
    recorded as real. The referee times a seed the worker never saw and scores
    that; the gap between the two is the tell."""
    box = make_box(speedup=2985.0, overfit_inputs=("er1000_005", "gn800"))
    ref = Referee(box, config, draw_seed=lambda: 777)
    ref.setup()
    m = ref.measure(PRECOMPUTE)
    assert m.tests_pass and m.result_matches is True
    assert m.overfit == ("er1000_005", "gn800") and not m.clears_noise
    fitted, honest = _first(m, "er1000_005"), _first(m, "er1000_001")
    assert fitted.seed == 777 and fitted.speedup == pytest.approx(1.0)
    assert fitted.shown_speedup == pytest.approx(2985.0) and fitted.overfit
    assert any("overfit" in e and "2985.0x" in e for e in fitted.errors)
    assert honest.speedup == pytest.approx(2985.0) and not honest.overfit
    # The geomean is over held out ratios, so the two recognised inputs count as 1.0x.
    assert m.speedup is not None and m.speedup == pytest.approx(2985.0 ** (3 / 5))
    assert m.to_dict()["overfit"] == ["er1000_005", "gn800"]


def test_a_patch_wrong_on_the_held_out_instance_computed_the_wrong_thing(
    config: RunConfig,
) -> None:
    box = make_box(speedup=3.0, held_out_wrong=("gn800",))
    m = _referee(box, config).measure(PRECOMPUTE)
    assert m.tests_pass and m.result_matches is False and not m.clears_noise
    i = _first(m, "gn800")
    assert i.base_fp == i.patched_fp and i.held_out_base_fp != i.held_out_patched_fp
    assert i.result_matches is False and _first(m, "er1000_005").result_matches is True
    assert i.speedup is not None  # still timed and recorded in full


def test_measure_refuses_a_setup_that_does_not_read_seed_by_name(config: RunConfig) -> None:
    first, *rest = config.target.inputs
    point = replace(first, setup=first.setup.replace(" + SEED", ""))
    one_point = replace(config, target=replace(config.target, inputs=(point, *rest)))
    assert one_point.target.unseeded == (first.name,)
    box = make_box()
    ref = _referee(box, one_point)
    with pytest.raises(ValueError, match=f"SEED: {first.name}"):
        ref.measure(PRECOMPUTE)
    assert not any("apply_patch.py" in c for c in box.commands)


def test_null_pairs_time_a_held_out_seed_per_round(config: RunConfig) -> None:
    """A floor is the family's, not one instance's: each round of null pairs
    builds the inputs at a seed of its own, the way a measurement would."""
    drawn = iter([11, 22])
    box = make_box()
    ref = Referee(box, config, draw_seed=lambda: next(drawn))
    ref.setup()
    out = ref.null_pairs(rounds=2)
    for pairs in out.values():
        assert [p.seed for p in pairs] == [11] * 6 + [22] * 6
    verifies = [c for c in box.commands if "--verify" in c]
    assert all(arg(c, "--seed") == "0" for c in verifies)
