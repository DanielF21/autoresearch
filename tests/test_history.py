from pathlib import Path

import pytest

from autoresearch.history import (
    HistoryError,
    RunPaths,
    acquire_lock,
    append_round,
    best_ratio,
    last_completed_round,
    load_history,
    next_attempt_number,
    read_boxes,
    read_rounds,
    release_lock,
    write_attempt,
    write_boxes,
    write_measurement,
)
from autoresearch.types import (
    AttemptRef,
    InputTiming,
    Measurement,
    Prediction,
    RoundRecord,
    StopReason,
    SuiteResult,
    Usage,
    WorkerOutput,
)


def _output(patch: str | None = "diff --git a/x b/x\n") -> WorkerOutput:
    return WorkerOutput(
        patch=patch,
        prediction=Prediction(1.1, 0.7) if patch else None,
        rationale="tried a thing",
        stop_reason=StopReason.SUBMITTED if patch else StopReason.MAX_TURNS,
        turns=12,
        usage=Usage(1000, 500, 100, 50),
        wall_s=42.0,
    )


def test_write_then_load_round_trips_every_field(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    ref = AttemptRef(number=1, round=1, worker=0)
    write_attempt(paths, ref, "abc123", {"prompt_hash": "p"}, _output(), '{"turn": 1}\n')
    (h,) = load_history(paths)
    assert h.ref == ref
    assert h.base_sha == "abc123"
    assert h.patch == "diff --git a/x b/x\n"
    assert h.prediction == Prediction(1.1, 0.7)
    assert h.rationale == "tried a thing"
    assert h.stop_reason == StopReason.SUBMITTED
    assert h.usage == Usage(1000, 500, 100, 50)
    assert h.wall_s == 42.0
    assert h.measurement is None
    assert h.skipped == "" and h.duplicate_of == ""
    assert not h.clears_noise


def _timing(ratio: float | None, name: str = "bench") -> InputTiming:
    return InputTiming(name=name, noise_floor=1.0106, base_fp="a", patched_fp="a", speedup=ratio)


def test_measurement_is_added_once_and_appears_in_history(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    ref = AttemptRef(1, 1, 0)
    write_attempt(paths, ref, "abc", {}, _output(), "", duplicate_of="0000")
    m = Measurement(applied=True, inputs=(_timing(1.002),))
    write_measurement(paths, ref, m)
    h = load_history(paths)[0]
    assert h.measurement == m and h.duplicate_of == "0000"
    with pytest.raises(HistoryError):
        write_measurement(paths, ref, m)


def test_best_ratio_counts_only_real_speedups() -> None:
    def att(n: int, ratio: float, ok: bool) -> object:
        m = Measurement(
            applied=True,
            tests=(SuiteResult("module", 1, 0, 0, 1, ok), SuiteResult("full", 1, 0, 0, 1, ok)),
            inputs=(_timing(ratio),),
        )
        return _attempt(n, m)

    from autoresearch.types import Attempt

    def _attempt(n: int, m: Measurement) -> Attempt:
        return Attempt(
            AttemptRef(n, n, 0), "s", "d", None, "", StopReason.SUBMITTED, Usage(), 1.0, m
        )

    history = (att(1, 1.05, True), att(2, 1.30, False), att(3, 1.08, True), att(4, 1.001, True))
    assert best_ratio(history) == (1.08, 3)  # type: ignore[arg-type]
    assert best_ratio(()) == (None, None)


def test_attempts_are_write_once(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    ref = AttemptRef(1, 1, 0)
    write_attempt(paths, ref, "abc", {}, _output(), "")
    with pytest.raises(HistoryError):
        write_attempt(paths, ref, "abc", {}, _output(), "")


def test_history_is_ordered_by_number_and_next_number_follows(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    assert next_attempt_number(paths) == 1
    for n in (2, 1, 3):
        write_attempt(
            paths, AttemptRef(n, 1, n - 1), "abc", {}, _output(None), "", skipped="no_patch"
        )
    assert [h.ref.number for h in load_history(paths)] == [1, 2, 3]
    assert next_attempt_number(paths) == 4


def test_no_patch_attempt_has_no_patch_file(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    ref = AttemptRef(1, 1, 0)
    d = write_attempt(paths, ref, "abc", {}, _output(None), "", skipped="no_patch")
    assert not (d / "patch.diff").exists()
    assert load_history(paths)[0].patch is None
    assert load_history(paths)[0].skipped == "no_patch"


def test_rounds_append_and_read(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    assert last_completed_round(paths) == 0
    rec = RoundRecord(1, (1,), "a", (1,), (1,), 1.05, 10.0, 20.0, Usage(), (), "t")
    append_round(paths, rec)
    append_round(paths, RoundRecord(2, (2,), "a", (), (), 1.05, 10.0, 20.0, Usage(), ("x",), "t"))
    assert read_rounds(paths)[0] == rec
    assert last_completed_round(paths) == 2


def test_boxes_round_trip(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    assert read_boxes(paths) == {}
    write_boxes(paths, {"referee:0": "sb_1"})
    assert read_boxes(paths) == {"referee:0": "sb_1"}


def test_lock_refuses_a_second_orchestrator(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    acquire_lock(paths)
    with pytest.raises(HistoryError, match="locked"):
        acquire_lock(paths)
    release_lock(paths)
    acquire_lock(paths)
