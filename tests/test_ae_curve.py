"""The head to head curve, read from run directories."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from alphaevolve import curve
from alphaevolve.cli import main
from autoresearch import history
from autoresearch.types import (
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    RoundRecord,
    SuiteResult,
    Usage,
)
from tests.helpers import BASE_SHA, diff_for, submitted


def _measurement(speedup: float) -> Measurement:
    pair = PairTiming(0, "base_first", 0, 1.0, 1.0 / speedup, contaminated=False)
    return Measurement(
        applied=True,
        tests=(SuiteResult("module", 1, 0, 0, 1.0, True), SuiteResult("full", 1, 0, 0, 1.0, True)),
        inputs=(InputTiming("dense", 1.01, "a", "a", (pair,), speedup),),
    )


def write_round(
    paths: history.RunPaths,
    round_no: int,
    numbers: list[int],
    usage: Usage,
    speedups: list[float],
) -> None:
    for n, s in zip(numbers, speedups, strict=True):
        ref = AttemptRef(n, round_no, n - 1)
        history.write_attempt(paths, ref, BASE_SHA, {}, submitted(diff_for(str(n))), "")
        history.write_measurement(paths, ref, _measurement(s))
    history.append_round(
        paths,
        RoundRecord(
            round=round_no,
            attempt_numbers=tuple(numbers),
            base_sha=BASE_SHA,
            measured_numbers=tuple(numbers),
            clears_noise_numbers=(),
            best_ratio_so_far=None,
            worker_wall_s=1.0,
            referee_wall_s=1.0,
            usage=usage,
            errors=(),
            finished_at="2026-09-14T00:00:00+00:00",
        ),
    )


def _run(root: Path, rounds: list[tuple[Usage, list[float]]]) -> Path:
    paths = history.RunPaths(root)
    paths.attempts.mkdir(parents=True)
    number = 1
    for i, (usage, speedups) in enumerate(rounds, 1):
        numbers = list(range(number, number + len(speedups)))
        write_round(paths, i, numbers, usage, speedups)
        number += len(speedups)
    return root


def test_points_are_cumulative_tokens_and_the_best_real_speedup(tmp_path: Path) -> None:
    run = _run(
        tmp_path / "a",
        [(Usage(1000, 900, 50, 10), [1.0, 1.2]), (Usage(2000, 1500, 100, 0), [3.0, 1.0])],
    )
    points = curve.points(run)
    assert [(p.round, p.tokens, p.uncached, p.measurements) for p in points] == [
        (0, 0, 0, 0),
        (1, 1050, 150, 2),
        (2, 3150, 750, 4),
    ]
    assert points[0].best is None
    assert points[1].best == pytest.approx(1.2) and points[2].best == pytest.approx(3.0)
    assert curve.cut(points, 2000) == points[:2]


def test_the_curve_command_cuts_both_arms_at_the_smaller_total(tmp_path: Path) -> None:
    a = _run(tmp_path / "harness", [(Usage(1000, 0, 0, 0), [1.2]), (Usage(1000, 0, 0, 0), [2.0])])
    b = _run(tmp_path / "evolve", [(Usage(1500, 0, 0, 0), [1.5])])
    out = tmp_path / "out"
    assert main(["curve", str(a), str(b), "--out", str(out)]) == 0
    rows = list(csv.DictReader((out / "curve.csv").open()))
    assert [(r["label"], r["round"], r["best"]) for r in rows] == [
        ("harness", "0", ""),
        ("harness", "1", "1.200000"),
        ("evolve", "0", ""),
        ("evolve", "1", "1.500000"),
    ]
    assert (out / "curve.png").stat().st_size > 0
    assert main(["curve", str(a), "--label", "x", "--label", "y", "--out", str(out)]) == 2
