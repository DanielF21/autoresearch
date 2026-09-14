import math
from dataclasses import replace

import pytest

from autoresearch.types import (
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    Prediction,
    Provenance,
    RoundRecord,
    SuiteResult,
    Usage,
)

FLOOR = 1.0106


def _pair(i: int, base: float, pat: float, contaminated: bool = False) -> PairTiming:
    return PairTiming(
        index=i,
        order="base_first",
        hash_seed=i,
        base_s=base,
        patched_s=pat,
        contaminated=contaminated,
        reasons=("steal",) if contaminated else (),
    )


def _input(name: str, speedup: float | None, floor: float = FLOOR) -> InputTiming:
    return InputTiming(
        name=name,
        noise_floor=floor,
        base_fp="a",
        patched_fp="a",
        pairs=(_pair(0, 1.0, 1.0 / speedup),) if speedup else (),
        speedup=speedup,
    )


def _suite(scope: str, ok: bool = True) -> SuiteResult:
    return SuiteResult(scope, 10, 0 if ok else 1, 0, 1.5, ok)


def test_attempt_ref_dirname_is_zero_padded() -> None:
    assert AttemptRef(number=7, round=3, worker=0).dirname == "0007"


def test_usage_adds_fieldwise() -> None:
    a = Usage(prompt_tokens=10, cached_tokens=4, completion_tokens=2, reasoning_tokens=1)
    b = Usage(prompt_tokens=5, cached_tokens=5, completion_tokens=3, reasoning_tokens=0)
    assert a + b == Usage(15, 9, 5, 1)
    assert Usage.from_dict((a + b).to_dict()) == a + b


def test_pair_ratio_is_base_over_patched() -> None:
    assert _pair(0, 2.0, 1.0).ratio == 2.0


def test_a_pair_records_what_each_launch_paid_before_its_first_call() -> None:
    """Records written before the guest reported its process age have no such
    fields, and a machine without /proc reports None. Both must round trip."""
    with_fixed = replace(_pair(0, 2.0, 1.0), base_fixed_s=0.31, patched_fixed_s=0.29)
    assert PairTiming.from_dict(with_fixed.to_dict()) == with_fixed
    old = _pair(0, 2.0, 1.0).to_dict()
    del old["base_fixed_s"], old["patched_fixed_s"]
    assert PairTiming.from_dict(old) == _pair(0, 2.0, 1.0)
    assert PairTiming.from_dict(old).base_fixed_s is None


def test_measurement_round_trip() -> None:
    m = Measurement(
        applied=True,
        scope_violations=(),
        tests=(_suite("module"), _suite("full")),
        canary_s=0.51,
        inputs=(
            InputTiming(
                name="dense",
                noise_floor=FLOOR,
                base_fp="a",
                patched_fp="a",
                pairs=(_pair(0, 1.0, 0.95), _pair(1, 1.0, 0.96, contaminated=True)),
                speedup=1.0526,
                retried=True,
                errors=("one contaminated pair",),
                retries=2,
            ),
            _input("sparse", 1.5),
        ),
        errors=("timing skipped: patched tree cannot run",),
        provenance=Provenance(box_id="sb_x", cpu_flag_hash="abc"),
        wall_s=200.0,
    )
    assert Measurement.from_dict(m.to_dict()) == m
    assert m.tests_pass and m.result_matches is True and m.clears_noise
    assert m.inputs[0].retried and not m.inputs[1].retried
    assert m.inputs[0].retries == 2 and m.inputs[1].retries == 0
    # A record from before the count was kept says only that a retry happened.
    old = m.inputs[0].to_dict()
    del old["retries"]
    assert InputTiming.from_dict(old).retries == 1


def test_the_recorded_speedup_is_the_geometric_mean_over_inputs() -> None:
    m = Measurement(inputs=(_input("a", 128.0), _input("b", 2.0)))
    assert m.speedup == pytest.approx(math.sqrt(128.0 * 2.0))
    assert m.worst_speedup == 2.0
    # Breadth is worth the input count: doubling every input doubles the mean,
    # doubling one of five multiplies it by the fifth root of two.
    flat = Measurement(inputs=tuple(_input(str(i), 2.0) for i in range(5)))
    spike = Measurement(inputs=(_input("0", 4.0), *(_input(str(i), 2.0) for i in range(1, 5))))
    assert flat.speedup == pytest.approx(2.0)
    assert spike.speedup == pytest.approx(2.0 * 2 ** (1 / 5))


def test_an_input_with_no_timing_is_left_out_of_the_mean_and_named() -> None:
    m = Measurement(inputs=(_input("a", 4.0), _input("b", None)))
    assert m.speedup == 4.0 and m.worst_speedup == 4.0
    assert m.untimed == ("b",) and m.to_dict()["untimed"] == ["b"]
    assert Measurement(inputs=(_input("b", None),)).speedup is None


def test_a_record_written_before_instruction_counting_was_retired_still_loads() -> None:
    # Runs under runs/ carry an "ir" block. The field is gone; the records are not
    # rewritten, so from_dict has to read past the key rather than fail on it.
    old = Measurement(applied=True, inputs=(_input("x", 1.5),)).to_dict()
    old["ir"] = {"base": 5801995203, "patched": 610518708, "delta_pct": -89.48}
    m = Measurement.from_dict(old)
    assert m.speedup == 1.5 and not hasattr(m, "ir")


def test_a_record_written_before_the_referee_timed_more_than_one_input_still_loads() -> None:
    # The flat shape: pairs, speedup, base_fp and noise_floor at the top level
    # with no name for the one input. artifacts/generality.md.
    old = {
        "noise_floor": 1.0106,
        "applied": True,
        "tests": [_suite("module").to_dict(), _suite("full").to_dict()],
        "base_fp": "a",
        "patched_fp": "a",
        "pairs": [_pair(0, 1.0, 0.5).to_dict()],
        "speedup": 2.0,
        "wall_s": 130.0,
    }
    m = Measurement.from_dict(old)
    assert len(m.inputs) == 1
    assert m.inputs[0].name == "benchmark" and m.inputs[0].noise_floor == 1.0106
    assert m.speedup == 2.0 and m.worst_speedup == 2.0
    assert m.clears_noise and not m.regressions and len(m.inputs[0].pairs) == 1


def test_measurement_minimal_round_trip() -> None:
    m = Measurement(apply_error="does not apply")
    assert Measurement.from_dict(m.to_dict()) == m
    assert not m.applied and not m.tests_pass and m.result_matches is None and not m.clears_noise
    assert m.speedup is None and m.worst_speedup is None and m.regressions == ()


def test_clears_noise_requires_every_condition() -> None:
    good = Measurement(
        applied=True,
        tests=(_suite("module"), _suite("full")),
        inputs=(_input("dense", 1.02), _input("sparse", 1.02)),
    )
    assert good.clears_noise

    both_flat = replace(good, inputs=(_input("dense", 1.01), _input("sparse", 1.01)))
    assert not both_flat.clears_noise  # neither input clears its own floor
    assert not replace(good, inputs=(_input("dense", None), _input("sparse", None))).clears_noise
    # One untimed input disqualifies the whole measurement, however good the rest:
    # the mean over what was timed says nothing about what was not.
    one_untimed = replace(good, inputs=(_input("dense", 30.0), _input("sparse", None)))
    assert one_untimed.speedup == pytest.approx(30.0) and one_untimed.untimed == ("sparse",)
    assert not one_untimed.clears_noise
    assert not replace(good, tests=(_suite("module"), _suite("full", ok=False))).clears_noise
    assert not replace(good, tests=(_suite("module"),)).clears_noise  # full suite never ran
    assert not replace(good, scope_violations=("tests edited",)).clears_noise
    assert not replace(good, applied=False).clears_noise
    # A fingerprint that differs on any single input fails the whole measurement.
    odd = replace(_input("sparse", 1.02), patched_fp="b")
    assert not replace(good, inputs=(_input("dense", 1.02), odd)).clears_noise
    blank = replace(_input("sparse", 1.02), patched_fp="")
    assert replace(good, inputs=(_input("dense", 1.02), blank)).result_matches is None
    # A cached result on repeated calls is named and never a real speedup.
    cached = replace(_input("sparse", 400.0), memoized=True)
    flagged = replace(good, inputs=(_input("dense", 1.02), cached))
    assert flagged.memoized == ("sparse",) and not flagged.clears_noise
    assert Measurement.from_dict(flagged.to_dict()) == flagged
    assert flagged.to_dict()["memoized"] == ["sparse"]
    # A record from before the flag existed reads as not memoized.
    old_input = _input("sparse", 1.02).to_dict()
    del old_input["memoized"]
    assert InputTiming.from_dict(old_input).memoized is False


def test_a_patch_that_recognises_the_shown_instance_is_overfit_not_a_speedup() -> None:
    """pycodestyle_p3_w16 attempt 0233: 2985x on the six instances the worker was
    shown, by string equality against their regenerated text. The score is the
    held out instance's ratio; the shown one is kept beside it, and the flag
    keeps the attempt out of the leader set."""
    good = Measurement(
        applied=True,
        tests=(_suite("module"), _suite("full")),
        inputs=(_input("dense", 1.02), _input("sparse", 1.02)),
    )
    recognised = replace(
        _input("sparse", 1.0),
        seed=123456789,
        shown_pairs=(_pair(0, 1.0, 1 / 2985.0),),
        shown_speedup=2985.0,
        held_out_base_fp="h",
        held_out_patched_fp="h",
        overfit=True,
    )
    flagged = replace(good, inputs=(_input("dense", 1.02), recognised))
    assert flagged.overfit == ("sparse",) and not flagged.clears_noise
    assert flagged.speedup == pytest.approx(math.sqrt(1.02))  # the held out ratios
    assert Measurement.from_dict(flagged.to_dict()) == flagged
    assert flagged.to_dict()["overfit"] == ["sparse"]
    assert flagged.to_dict()["inputs"][1]["shown_speedup"] == 2985.0
    # A record from before the held out timing reads as a shown only score.
    old_input = _input("sparse", 1.02).to_dict()
    for key in ("overfit", "seed", "shown_pairs", "shown_speedup", "held_out_base_fp"):
        del old_input[key]
    old = InputTiming.from_dict(old_input)
    assert old.overfit is False and old.seed == 0 and old.shown_pairs == ()
    assert old.result_matches is True


def test_result_matches_covers_the_held_out_instance_when_it_was_verified() -> None:
    """A patch right on seed 0 and wrong on the seed it never saw computed the
    wrong thing. An older record with no held out fingerprints compares the
    shown ones alone."""
    shown_only = _input("a", 1.02)
    assert shown_only.result_matches is True
    both = replace(shown_only, held_out_base_fp="h1", held_out_patched_fp="h1")
    assert both.result_matches is True
    wrong_held_out = replace(shown_only, held_out_base_fp="h1", held_out_patched_fp="h2")
    assert wrong_held_out.result_matches is False
    wrong_shown = replace(both, patched_fp="b")
    assert wrong_shown.result_matches is False
    assert replace(shown_only, patched_fp="").result_matches is None
    # A held out verify that failed on one tree leaves one side blank: not a match.
    assert replace(shown_only, held_out_base_fp="h1").result_matches is False


def test_a_failed_suite_names_its_tests_and_an_old_record_has_none() -> None:
    tail = (
        "=== short test summary info ===\n"
        "FAILED tests/test_unit.py::test_dates - AssertionError: 2026\n"
        "ERROR tests/test_io.py::test_read\n"
        "1 failed, 1 error, 2135 passed in 21s\n"
    )
    broken = replace(_suite("full", ok=False), failures=tail)
    assert broken.failed_tests == ("tests/test_unit.py::test_dates", "tests/test_io.py::test_read")
    assert SuiteResult.from_dict(broken.to_dict()) == broken
    old = _suite("full").to_dict()
    del old["failures"]
    assert (
        SuiteResult.from_dict(old).failures == "" and SuiteResult.from_dict(old).failed_tests == ()
    )


def test_a_pair_records_which_instance_it_timed() -> None:
    held_out = replace(_pair(0, 2.0, 1.0), seed=987654321)
    assert PairTiming.from_dict(held_out.to_dict()) == held_out
    old = _pair(0, 2.0, 1.0).to_dict()
    del old["seed"]
    assert PairTiming.from_dict(old).seed == 0


def test_a_patch_slower_on_any_input_is_not_a_real_speedup() -> None:
    """The whole reason the referee times a set. artifacts/generality.md.

    Patch 0010's shape: enormous on the dense input, half speed on the sparse
    one. Its geometric mean is 5.5x and it must still be rejected.
    """
    m = Measurement(
        applied=True,
        tests=(_suite("module"), _suite("full")),
        inputs=(_input("dense", 139.69), _input("sparse", 0.22)),
    )
    assert m.speedup is not None and m.speedup > 5.0
    assert m.regressions == ("sparse",)
    assert not m.clears_noise
    # Guarding the fast path recovers it: still huge on dense, now merely
    # neutral on sparse rather than slower.
    guarded = Measurement(
        applied=True,
        tests=(_suite("module"), _suite("full")),
        inputs=(_input("dense", 128.57), _input("sparse", 0.995)),
    )
    assert guarded.regressions == () and guarded.clears_noise


def test_a_regression_is_the_mirror_of_the_floor() -> None:
    floor = 1.25
    assert not _input("x", 1.0 / 1.24, floor).regresses  # inside the band
    assert _input("x", 1.0 / 1.26, floor).regresses
    assert _input("x", 1.26, floor).improves
    assert not _input("x", 1.24, floor).improves


def test_prediction_round_trip_with_and_without_confidence() -> None:
    for p in (Prediction(1.2), Prediction(1.2, 0.5)):
        assert Prediction.from_dict(p.to_dict()) == p


def test_round_record_round_trip() -> None:
    rec = RoundRecord(
        round=2,
        attempt_numbers=(3, 4),
        base_sha="a",
        measured_numbers=(3, 4),
        clears_noise_numbers=(4,),
        best_ratio_so_far=1.07,
        worker_wall_s=100.0,
        referee_wall_s=200.0,
        usage=Usage(1, 2, 3, 4),
        errors=(),
        finished_at="2026-09-11T00:00:00Z",
    )
    assert RoundRecord.from_dict(rec.to_dict()) == rec
    none = RoundRecord(1, (1,), "a", (), (), None, 0.0, 0.0, Usage(), (), "t")
    assert RoundRecord.from_dict(none.to_dict()) == none
