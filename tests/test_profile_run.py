"""The time profiler: how a transcript is split, and what the referee model claims."""

import json
from dataclasses import replace
from pathlib import Path

import profile_run
import pytest

from autoresearch.types import (
    Attempt,
    AttemptRef,
    InputTiming,
    PairTiming,
    Prediction,
    StopReason,
    SuiteResult,
    Usage,
)
from tests.helpers import BASE_SHA
from tests.test_analyze import measurement, pair, timing

ROOT = Path(__file__).parent.parent


def transcript(*records: dict[str, object]) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


def attempt(n: int, wall_s: float) -> Attempt:
    return Attempt(
        ref=AttemptRef(n, n, 0),
        base_sha=BASE_SHA,
        patch="diff",
        prediction=Prediction(1.0),
        rationale="r",
        stop_reason=StopReason.SUBMITTED,
        usage=Usage(),
        wall_s=wall_s,
        measurement=None,
    )


def test_the_worker_phases_are_split_at_the_recorded_timestamps() -> None:
    # box at t=100, first model call took 2s and its record lands at t=115, so
    # the call began at 113 and box prepare ran 100 -> 113.
    t = transcript(
        {"kind": "box", "t": 100.0, "box_id": "sb_x"},
        {"kind": "model", "t": 115.0, "latency_s": 2.0, "turn": 1, "tool_calls": []},
        {"kind": "tool", "t": 120.0, "name": "shell", "turn": 1},
        {"kind": "model", "t": 124.0, "latency_s": 1.0, "turn": 2, "tool_calls": []},
        {"kind": "tool", "t": 130.0, "name": "run_tests", "turn": 2},
        {"kind": "end", "t": 131.0, "turns": 2, "stop": "submitted"},
    )
    p = profile_run.worker_profile(attempt(1, wall_s=32.0), t)

    assert p.turns == 2
    assert p.parts["box create"] == 1.0  # start is 131 - 32 = 99, box at 100
    assert p.parts["box prepare"] == 13.0  # 100 -> 113
    assert p.parts["model"] == 3.0
    assert p.tools == {"shell": 5.0, "run_tests": 6.0}
    assert p.parts["tools"] == 11.0
    assert sum(p.parts.values()) == 32.0  # every second is attributed


def test_the_residual_lands_in_harness_and_never_goes_negative() -> None:
    t = transcript(
        {"kind": "box", "t": 100.0, "box_id": "sb_x"},
        {"kind": "model", "t": 102.0, "latency_s": 1.0, "turn": 1, "tool_calls": []},
        {"kind": "end", "t": 110.0, "turns": 1, "stop": "submitted"},
    )
    p = profile_run.worker_profile(attempt(1, wall_s=20.0), t)
    assert p.parts["harness"] > 0
    assert sum(p.parts.values()) == 20.0

    # A wall clock shorter than the timestamps span must not draw a negative bar.
    short = profile_run.worker_profile(attempt(1, wall_s=1.0), t)
    assert all(v >= 0 for v in short.parts.values()), short.parts


def test_an_attempt_with_no_transcript_is_all_harness() -> None:
    p = profile_run.worker_profile(attempt(1, wall_s=50.0), "")
    assert p.parts == {"harness": 50.0}


def test_the_slowest_model_call_and_tool_are_kept() -> None:
    t = transcript(
        {"kind": "box", "t": 0.0, "box_id": "sb_x"},
        {"kind": "model", "t": 10.0, "latency_s": 10.0, "turn": 1, "tool_calls": []},
        {"kind": "tool", "t": 11.0, "name": "shell", "turn": 1},
        {"kind": "model", "t": 14.0, "latency_s": 3.0, "turn": 2, "tool_calls": []},
        {"kind": "tool", "t": 100.0, "name": "run_tests", "turn": 2},
        {"kind": "end", "t": 100.0, "turns": 2, "stop": "submitted"},
    )
    p = profile_run.worker_profile(attempt(1, wall_s=100.0), t)
    assert p.slowest_model_s == 10.0
    assert p.slowest_tool == ("run_tests", 86.0)


FACTS = profile_run.RunFacts(width=4, repeats_per_launch=7)


def test_the_referee_model_charges_one_startup_per_launch() -> None:
    m = measurement(0.1, 10.0, pairs=tuple(pair(i, 1.0, 0.1) for i in range(6)))
    p = profile_run.referee_profile(1, m, FACTS)

    # 6 launches per tree, each one fixed startup plus repeats_per_launch bodies.
    repeats = FACTS.repeats_per_launch
    assert p.parts["timing base"] == 6 * (profile_run.LEGACY_LAUNCH_FIXED_S + repeats * 1.0)
    assert p.parts["timing patched"] == 6 * (profile_run.LEGACY_LAUNCH_FIXED_S + repeats * 0.1)
    assert p.launches == 2 * 6 + 2 + 1  # timing, two verify, one canary


def test_a_recorded_fixed_cost_replaces_the_modelled_one() -> None:
    """Since the guest reports its process age, the launch overhead is measured
    per launch and the networkx constant is only for records that predate it."""
    pairs = tuple(
        replace(pair(i, 1.0, 0.1), base_fixed_s=0.5, patched_fixed_s=0.7) for i in range(6)
    )
    m = measurement(0.1, 10.0, pairs=pairs)
    p = profile_run.referee_profile(1, m, FACTS)
    repeats = FACTS.repeats_per_launch
    assert p.parts["timing base"] == pytest.approx(6 * (0.5 + repeats * 1.0))
    assert p.parts["timing patched"] == pytest.approx(6 * (0.7 + repeats * 0.1))
    assert p.parts["verify"] == pytest.approx(0.5 + 0.7 + 1.0 + 0.1)
    # A mix: one launch recorded, the other not, each attributed its own way.
    mixed = tuple(replace(x, patched_fixed_s=None) for x in pairs)
    q = profile_run.referee_profile(1, measurement(0.1, 10.0, pairs=mixed), FACTS)
    assert q.parts["timing patched"] == pytest.approx(
        6 * (profile_run.LEGACY_LAUNCH_FIXED_S + repeats * 0.1)
    )


def test_the_launch_count_and_the_timing_bands_are_sums_over_inputs() -> None:
    two = measurement(
        0.0, 0.0, inputs=(timing("dense", 10.0, patched_s=0.1), timing("sparse", 2.0))
    )
    p = profile_run.referee_profile(1, two, FACTS)
    one = profile_run.referee_profile(1, measurement(0.1, 10.0), FACTS)
    assert p.launches == 2 * one.launches - 1  # the canary is charged once
    assert p.parts["timing base"] == pytest.approx(2 * one.parts["timing base"])


def test_a_retried_input_is_named_and_its_launches_are_counted_twice() -> None:
    """The discarded first pass is not in the record. Attempt 0009 of t1_w4b was
    exactly this, and showed up only as 73.7s of unexplained residual."""
    retimed = replace(timing("dense", 10.0, patched_s=0.1), retried=True)
    m = measurement(0.0, 0.0, inputs=(retimed,))
    p = profile_run.referee_profile(1, m, FACTS)
    assert p.retried == ("dense",)
    assert p.launches == 2 * 2 * 6 + 2 + 1
    assert profile_run.referee_profile(1, measurement(0.1, 10.0), FACTS).retried == ()


def test_everything_the_referee_record_does_not_explain_lands_in_one_named_block() -> None:
    m = measurement(0.1, 10.0, pairs=tuple(pair(i, 1.0, 0.1) for i in range(6)))
    m = replace(m, wall_s=5000.0)
    p = profile_run.referee_profile(1, m, FACTS)
    assert sum(p.parts.values()) == 5000.0
    assert p.parts["setup and cleanup"] > 4000


def test_a_referee_that_recorded_an_error_is_flagged() -> None:
    clean = measurement(0.1, 10.0)
    assert profile_run.referee_profile(1, clean, FACTS).errors == 0
    broke = replace(clean, errors=("timing: guest died", "cleanup"))
    assert profile_run.referee_profile(1, broke, FACTS).errors == 2
    # An error recorded against one input counts too.
    per_input = measurement(0.0, 0.0, inputs=(replace(timing("d", 2.0), errors=("x",)),))
    assert profile_run.referee_profile(1, per_input, FACTS).errors == 1


def test_run_facts_reads_a_config_the_strict_parser_would_refuse(tmp_path: Path) -> None:
    """An old run's frozen config cannot load, and must still profile."""
    old = tmp_path / "config.toml"
    old.write_text(
        '[run]\nrun_id = "x"\nwidth = 4\nrounds = 3\n\n'
        '[target]\ngraph = "nx.g()"\ncall = "nx.c(G)"\n\n'
        "[referee]\npairs = 6\nnoise_floor = 1.0106\nrepeats_per_launch = 7\n"
    )
    facts = profile_run.read_run_facts(old)
    assert facts.width == 4 and facts.repeats_per_launch == 7


def test_the_barrier_cost_is_the_wait_for_the_slowest() -> None:
    r = profile_run.RoundProfile(
        round=1,
        worker_wall_s=100.0,
        referee_wall_s=200.0,
        worker_each=[100.0, 40.0, 60.0, 80.0],
        referee_each=[200.0, 150.0, 150.0, 200.0],
    )
    assert r.wall_s == 300.0
    assert r.worker_idle_s == 0 + 60 + 40 + 20
    assert r.referee_idle_s == 0 + 50 + 50 + 0


def test_the_suite_result_and_pair_fixtures_still_match_the_types() -> None:
    # Guards the fixtures these tests are built on against a types.py change.
    assert isinstance(measurement(0.1, 2.0).tests[0], SuiteResult)
    assert isinstance(measurement(0.1, 2.0).inputs[0], InputTiming)
    assert isinstance(measurement(0.1, 2.0).inputs[0].pairs[0], PairTiming)
