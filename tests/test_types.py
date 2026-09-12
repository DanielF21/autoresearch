from autoresearch.types import (
    AttemptRef,
    Measurement,
    PairTiming,
    Prediction,
    Provenance,
    RoundRecord,
    SuiteResult,
    Usage,
)


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


def test_measurement_round_trip() -> None:
    m = Measurement(
        noise_floor=1.0106,
        applied=True,
        scope_violations=(),
        tests=(_suite("module"), _suite("full")),
        base_fp="a",
        patched_fp="a",
        canary_s=0.51,
        pairs=(_pair(0, 1.0, 0.95), _pair(1, 1.0, 0.96, contaminated=True)),
        speedup=1.0526,
        errors=("timing skipped: patched tree cannot run",),
        provenance=Provenance(box_id="sb_x", cpu_flag_hash="abc"),
        wall_s=200.0,
    )
    assert Measurement.from_dict(m.to_dict()) == m
    assert m.tests_pass and m.result_matches is True and m.clears_noise


def test_a_record_written_before_instruction_counting_was_retired_still_loads() -> None:
    # Runs under runs/ carry an "ir" block. The field is gone; the records are not
    # rewritten, so from_dict has to read past the key rather than fail on it.
    old = Measurement(noise_floor=1.0106, applied=True, speedup=1.5).to_dict()
    old["ir"] = {"base": 5801995203, "patched": 610518708, "delta_pct": -89.48}
    m = Measurement.from_dict(old)
    assert m.speedup == 1.5 and not hasattr(m, "ir")


def test_measurement_minimal_round_trip() -> None:
    m = Measurement(noise_floor=1.01, apply_error="does not apply")
    assert Measurement.from_dict(m.to_dict()) == m
    assert not m.applied and not m.tests_pass and m.result_matches is None and not m.clears_noise


def test_clears_noise_requires_every_condition() -> None:
    good = Measurement(
        noise_floor=1.0106,
        applied=True,
        tests=(_suite("module"), _suite("full")),
        base_fp="a",
        patched_fp="a",
        speedup=1.02,
    )
    assert good.clears_noise
    from dataclasses import replace

    assert not replace(good, speedup=1.01).clears_noise
    assert not replace(good, speedup=None).clears_noise
    assert not replace(good, tests=(_suite("module"), _suite("full", ok=False))).clears_noise
    assert not replace(good, tests=(_suite("module"),)).clears_noise  # full suite never ran
    assert not replace(good, patched_fp="b").clears_noise
    assert not replace(good, patched_fp="").clears_noise
    assert not replace(good, scope_violations=("tests edited",)).clears_noise
    assert not replace(good, applied=False).clears_noise


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
