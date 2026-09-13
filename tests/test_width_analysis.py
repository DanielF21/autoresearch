"""The width analysis loader: the record rule, function attribution, and the tables."""

import json
from pathlib import Path

from analysis.width import dims, load

from autoresearch import history
from autoresearch.types import (
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    Prediction,
    RoundRecord,
    StopReason,
    SuiteResult,
    Usage,
    WorkerOutput,
)
from tests.helpers import BASE_SHA

BASE = '''"""a module"""
import math


def slow(xs):
    total = 0
    for x in xs:
        total += x
    return total


class K:
    def method(self, y):
        return y * 2
'''

CONFIG = """[run]
run_id = "t_w2"
width = 2
rounds = 3

[worker]
model = "deepseek/deepseek-v4-pro-0813"
"""


def diff(old_start: int, added: tuple[str, ...], path: str = "pkg/mod.py") -> str:
    body = "\n".join(f"+{line}" for line in added)
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -{old_start},2 +{old_start},{2 + len(added)} @@\n context\n{body}\n context\n"
    )


def timing(name: str, speedup: float | None, floor: float = 1.01) -> InputTiming:
    pairs = tuple(
        PairTiming(i, "base_first", i, 1.0, 1.0 / (speedup or 1.0), contaminated=speedup is None)
        for i in range(6)
    )
    return InputTiming(name, floor, "a", "a", pairs, speedup)


def measurement(a: float | None, b: float | None) -> Measurement:
    return Measurement(
        applied=True,
        tests=(SuiteResult("module", 1, 0, 0, 1.0, True), SuiteResult("full", 1, 0, 0, 1.0, True)),
        inputs=(timing("dense", a), timing("sparse", b)),
    )


def write_run(root: Path) -> Path:
    """Width 2, three rounds, the third not completed. Attempt 4 has an untimed input."""
    run = root / "t_w2"
    run.mkdir()
    (run / "config.toml").write_text(CONFIG)
    paths = history.RunPaths(run)
    plan = [
        (
            1,
            1,
            0,
            diff(5, ("total = sum(xs)",)),
            "Replace the loop with sum. A bitmask.",
            2.0,
            2.0,
            [],
        ),
        (2, 1, 1, diff(13, ("return y + y",)), "numpy path", 1.5, 1.5, []),
        (3, 2, 0, diff(5, ("total = sum(xs)", "return total")), "", 2.5, 2.5, [1, 2]),
        (4, 2, 1, diff(5, ("total = math.fsum(xs)",)), "cache it", 9.0, None, [1, 2]),
        (5, 3, 0, diff(1, ("import os",)), "module level", 1.2, 1.2, [1, 2, 3, 4]),
    ]
    for number, rnd, worker, patch, rationale, a, b, seen in plan:
        ref = AttemptRef(number, rnd, worker)
        out = WorkerOutput(patch, Prediction(2.0), rationale, StopReason.SUBMITTED, 3, Usage(), 1.0)
        history.write_attempt(
            paths, ref, BASE_SHA, {"history_numbers": seen, "config_hash": "x"}, out, ""
        )
        history.write_measurement(paths, ref, measurement(a, b))
        if b is None:
            # Written by the referee before it required every input timed: the run
            # counted this attempt as a real speedup and showed it as the leader.
            path = paths.attempt(ref) / history.MEASUREMENT_JSON
            record = json.loads(path.read_text())
            record["clears_noise"] = True
            path.write_text(json.dumps(record))
    for rnd, numbers in ((1, (1, 2)), (2, (3, 4))):
        history.append_round(
            paths,
            RoundRecord(
                rnd, numbers, BASE_SHA, numbers, numbers, None, 10.0, 5.0, Usage(), (), "t"
            ),
        )
    return run


def base_source(path: str) -> str | None:
    return BASE if path == "pkg/mod.py" else None


def test_the_loader_keeps_completed_rounds_and_marks_records_strictly(tmp_path: Path) -> None:
    run = load.load_run(write_run(tmp_path), base_source)
    assert run.width == 2 and run.rounds_done == 2 and [r.number for r in run.rows] == [1, 2, 3, 4]
    by = {r.number: r for r in run.rows}
    # Attempt 4 cleared with the best geomean but one input never timed: the run counted
    # it, the analysis does not.
    assert not by[4].clears_noise and not by[4].complete  # the rule now refuses it too
    assert by[4].record_seen and not by[4].record
    assert [r.record for r in run.rows] == [True, False, True, False]
    assert by[3].record_at_time == 1 and by[3].overlap == 1.0 and by[3].builds_on_record
    assert by[4].record_at_time == 1 and by[4].overlap == 0.0


def test_hunks_map_to_the_enclosing_function_of_the_base_file() -> None:
    assert load.touched_functions(diff(5, ("x",)), base_source) == ("slow",)
    assert load.touched_functions(diff(13, ("x",)), base_source) == ("K.method",)
    assert load.touched_functions(diff(1, ("x",)), base_source) == ("pkg/mod.py:module",)
    assert load.touched_functions(diff(1, ("x",), "pkg/new.py"), base_source) == ("pkg/new.py:new",)


def test_strategy_classes_are_keyword_matches_over_the_rationale() -> None:
    assert load.strategies("Replace the loop with sum. A bitmask.") == ("bitset",)
    assert load.strategies("numpy path") == ("matrix",)
    assert load.strategies("") == ("none",)
    assert load.strategies("something else entirely") == ("other",)


def test_the_dimensions_write_a_table_and_a_figure_each(tmp_path: Path) -> None:
    run = load.load_run(write_run(tmp_path), base_source)
    out = tmp_path / "out"
    tables = dims.run_all([run], out)
    assert sorted(p.name for p in out.glob("d1_*")) == ["d1_headline.csv", "d1_headline.png"]
    d1 = tables["d1"]
    assert [t["best"] for t in d1] == [2.0, 2.5]
    assert [t["best_as_run_saw_it"] for t in d1] == [2.0, 9.0]
    assert tables["d2"][0]["records"] == 2 and tables["d2"][0]["incomplete_cleared"] == 1
    assert tables["d5"][0]["record"] == 1 and tables["d5"][0]["real speedup"] == 1
    assert len(tables["d7"]) == 2
    for n in range(1, 8):
        assert list(out.glob(f"d{n}_*.png"))
