"""The analysis script: which attempts count as progress, and what the table says."""

import analyze
import pytest

from autoresearch.types import (
    Attempt,
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    Prediction,
    StopReason,
    SuiteResult,
    Usage,
)
from tests.helpers import BASE_SHA

FLOOR = 1.0106
ONLY = "bench"  # the single input name most of these fixtures use


def pair(i: int, base_s: float, patched_s: float, contaminated: bool = False) -> PairTiming:
    return PairTiming(
        index=i,
        order="base_first" if i % 2 == 0 else "patched_first",
        hash_seed=i,
        base_s=base_s,
        patched_s=patched_s,
        contaminated=contaminated,
    )


def timing(
    name: str,
    speedup: float,
    patched_s: float = 0.1,
    pairs: tuple[PairTiming, ...] | None = None,
    floor: float = FLOOR,
) -> InputTiming:
    return InputTiming(
        name=name,
        noise_floor=floor,
        base_fp="a",
        patched_fp="a",
        speedup=speedup,
        pairs=pairs if pairs is not None else tuple(pair(i, 1.0, patched_s) for i in range(6)),
    )


def measurement(
    patched_s: float,
    speedup: float,
    tests_ok: bool = True,
    pairs: tuple[PairTiming, ...] | None = None,
    inputs: tuple[InputTiming, ...] | None = None,
) -> Measurement:
    return Measurement(
        applied=True,
        tests=(
            SuiteResult("module", 10, 0 if tests_ok else 1, 0, 1.0, tests_ok),
            SuiteResult("full", 100, 0 if tests_ok else 1, 0, 60.0, tests_ok),
        ),
        inputs=inputs
        if inputs is not None
        else (timing(ONLY, speedup, patched_s=patched_s, pairs=pairs),),
    )


def attempt(n: int, m: Measurement | None, rationale: str = "did a thing") -> Attempt:
    return Attempt(
        ref=AttemptRef(n, n, 0),
        base_sha=BASE_SHA,
        patch="diff" if m is not None else None,
        prediction=Prediction(1.0),
        rationale=rationale,
        stop_reason=StopReason.SUBMITTED,
        usage=Usage(),
        wall_s=1.0,
        measurement=m,
        skipped="" if m is not None else "no_patch",
    )


def test_the_running_best_only_moves_on_a_faster_attempt() -> None:
    rows = analyze.rows(
        (
            attempt(1, measurement(0.50, 2.0)),
            attempt(2, measurement(0.60, 1.7)),  # slower than 1, not a record
            attempt(3, measurement(0.25, 4.0)),
            attempt(4, measurement(0.25, 4.0)),  # equal, not an improvement
        )
    )
    assert [r.record for r in rows] == [True, False, True, False]
    assert [r.number for r in rows] == [1, 2, 3, 4]


def test_an_attempt_that_fails_its_tests_is_never_progress() -> None:
    # The largest speedup in the run, and it must not touch the line.
    rows = analyze.rows(
        (
            attempt(1, measurement(0.50, 2.0)),
            attempt(2, measurement(0.01, 100.0, tests_ok=False)),
        )
    )
    assert [r.record for r in rows] == [True, False]
    assert rows[1].clears is False


def test_an_attempt_below_the_noise_floor_is_never_progress() -> None:
    rows = analyze.rows((attempt(1, measurement(0.999, 1.001)),))
    assert rows[0].record is False and rows[0].clears is False


def test_an_attempt_with_no_patch_has_no_point_to_plot() -> None:
    rows = analyze.rows((attempt(1, None),))
    assert rows[0].speedup is None and rows[0].plotted is False and rows[0].record is False


def test_contaminated_pairs_are_left_out_of_the_per_input_wall_clock() -> None:
    # Same rule the referee uses for the median ratio, so the seconds reported
    # for an input and its recorded speedup are taken over the same launches.
    pairs = (pair(0, 1.0, 0.50, contaminated=True), pair(1, 1.0, 0.10), pair(2, 1.0, 0.10))
    assert analyze.median_patched_s(timing(ONLY, 2.0, pairs=pairs)) == 0.10


def test_wall_clock_falls_back_to_every_pair_when_none_is_clean() -> None:
    pairs = (pair(0, 1.0, 0.20, contaminated=True), pair(1, 1.0, 0.40, contaminated=True))
    t = timing(ONLY, 2.0, pairs=pairs)
    assert analyze.median_patched_s(t) == pytest.approx(0.30)
    assert analyze.median_base_s(t) == 1.0


def test_the_input_columns_are_every_input_seen_in_recording_order() -> None:
    m = measurement(0.0, 0.0, inputs=(timing("dense", 20.0), timing("sparse", 2.0)))
    assert analyze.input_names((attempt(1, m),)) == ["dense", "sparse"]
    # An old run has one unnamed input read back under a fixed name.
    assert analyze.input_names((attempt(1, measurement(0.1, 2.0)),)) == [ONLY]


def test_a_patch_slower_on_an_input_never_becomes_the_running_best() -> None:
    """Patch 0010's shape: a large mean carried by one input. artifacts/generality.md."""
    spiky = measurement(0.0, 0.0, inputs=(timing("dense", 139.69), timing("sparse", 0.22)))
    rows = analyze.rows((attempt(1, measurement(0.5, 2.0)), attempt(2, spiky)))
    assert rows[1].speedup is not None and rows[1].speedup > rows[0].speedup  # type: ignore[operator]
    assert rows[1].regressions == ("sparse",)
    assert [r.record for r in rows] == [True, False]


def test_the_label_is_cut_at_a_word_and_never_mid_word() -> None:
    long = "The directed clustering hot spot was rebuilding neighbour sets on every call"
    label = analyze.short_label(attempt(1, None, rationale=long))
    assert len(label) <= analyze.LABEL_CHARS + 1
    assert label.endswith("…") and not label[:-1].endswith(" ")
    assert long.startswith(label[:-1])


def test_the_label_is_the_first_sentence_only() -> None:
    a = attempt(1, None, rationale="Precomputed the sets. Then a second paragraph follows here.")
    assert analyze.short_label(a) == "Precomputed the sets"


def test_the_table_carries_the_baseline_row_and_marks_records() -> None:
    rows = analyze.rows((attempt(1, measurement(0.07, 19.5)),))
    text = analyze.render_table(rows, [ONLY])
    assert "0    baseline" in text and "1.000x" in text
    assert "19.500x" in text and text.rstrip().endswith("were slower on at least one input")
    assert "1 attempts, 1 marked * set a new best" in text


def test_the_table_has_a_column_per_input_and_counts_the_slower_ones() -> None:
    spiky = measurement(0.0, 0.0, inputs=(timing("dense", 139.69), timing("sparse", 0.22)))
    text = analyze.render_table(analyze.rows((attempt(1, spiky),)), ["dense", "sparse"])
    assert "dense" in text and "sparse" in text
    assert "139.690x" in text and "0.220x" in text
    assert "1 were slower on at least one input" in text
