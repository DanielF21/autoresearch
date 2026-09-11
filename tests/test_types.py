from autoresearch.types import (
    AttemptRef,
    IrCounts,
    PairTiming,
    Prediction,
    Provenance,
    RefereeResult,
    RoundRecord,
    SuiteResult,
    Usage,
    Verdict,
)


def _pair(i: int, inc: float, pat: float, contaminated: bool = False) -> PairTiming:
    return PairTiming(
        index=i,
        order="incumbent_first",
        hash_seed=i,
        incumbent_s=inc,
        patched_s=pat,
        contaminated=contaminated,
        reasons=("steal",) if contaminated else (),
    )


def test_attempt_ref_dirname_is_zero_padded() -> None:
    assert AttemptRef(number=7, round=3, worker=0).dirname == "0007"


def test_usage_adds_fieldwise() -> None:
    a = Usage(prompt_tokens=10, cached_tokens=4, completion_tokens=2, reasoning_tokens=1)
    b = Usage(prompt_tokens=5, cached_tokens=5, completion_tokens=3, reasoning_tokens=0)
    assert a + b == Usage(15, 9, 5, 1)
    assert Usage.from_dict((a + b).to_dict()) == a + b


def test_pair_ratio_is_incumbent_over_patched() -> None:
    assert _pair(0, 2.0, 1.0).ratio == 2.0


def test_ir_delta_pct() -> None:
    assert IrCounts(incumbent=1000, patched=900).delta_pct == -10.0


def test_referee_result_round_trip() -> None:
    result = RefereeResult(
        verdict=Verdict.ACCEPTED,
        reason="median 1.05 above 1.0106",
        threshold=1.0106,
        pairs=(_pair(0, 1.0, 0.95), _pair(1, 1.0, 0.96, contaminated=True)),
        median_ratio=1.0526,
        ir=IrCounts(100, 90),
        tests=(SuiteResult("module", 10, 0, 0, 1.5, True),),
        canary_s=0.51,
        provenance=Provenance(box_id="sb_x", cpu_flag_hash="abc"),
        wall_s=200.0,
    )
    back = RefereeResult.from_dict(result.to_dict())
    assert back == result


def test_referee_result_minimal_round_trip() -> None:
    result = RefereeResult(verdict=Verdict.REJECTED_SCOPE, reason="edited a test", threshold=1.01)
    assert RefereeResult.from_dict(result.to_dict()) == result


def test_prediction_round_trip_with_and_without_confidence() -> None:
    for p in (Prediction(1.2), Prediction(1.2, 0.5)):
        assert Prediction.from_dict(p.to_dict()) == p


def test_round_record_round_trip() -> None:
    rec = RoundRecord(
        round=2,
        attempt_numbers=(3, 4),
        incumbent_sha_before="a",
        incumbent_sha_after="b",
        accepted_numbers=(4,),
        worker_wall_s=100.0,
        referee_wall_s=200.0,
        usage=Usage(1, 2, 3, 4),
        errors=(),
        finished_at="2026-09-11T00:00:00Z",
    )
    assert RoundRecord.from_dict(rec.to_dict()) == rec
