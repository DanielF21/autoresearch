from pathlib import Path

from autoresearch import history
from autoresearch.orchestrator.status import compute_status
from autoresearch.types import (
    AttemptRef,
    Measurement,
    RoundRecord,
    SuiteResult,
    Usage,
)
from tests.helpers import BASE_SHA, diff_for, submitted

FLOOR = 1.0106


def _measurement(ratio: float | None, tests_ok: bool = True, matches: bool = True) -> Measurement:
    return Measurement(
        noise_floor=FLOOR,
        applied=True,
        tests=(
            SuiteResult("module", 10, 0 if tests_ok else 1, 0, 1.0, tests_ok),
            SuiteResult("full", 100, 0 if tests_ok else 1, 0, 60.0, tests_ok),
        ),
        base_fp="a",
        patched_fp="a" if matches else "b",
        speedup=ratio,
    )


def _attempt(
    paths: history.RunPaths, n: int, m: Measurement | None, duplicate_of: str = ""
) -> None:
    ref = AttemptRef(n, n, 0)
    out = submitted(diff_for(str(n)) if m is not None else None)
    history.write_attempt(
        paths,
        ref,
        BASE_SHA,
        {},
        out,
        "",
        skipped="" if m is not None else "no_patch",
        duplicate_of=duplicate_of,
    )
    if m is not None:
        history.write_measurement(paths, ref, m)


def test_status_totals(tmp_path: Path) -> None:
    paths = history.RunPaths(tmp_path)
    _attempt(paths, 1, _measurement(1.05))
    _attempt(paths, 2, _measurement(1.002))
    _attempt(paths, 3, _measurement(1.02))
    _attempt(paths, 4, None)
    _attempt(paths, 5, _measurement(1.30, tests_ok=False))  # fast but broken: not a real speedup
    _attempt(paths, 6, _measurement(1.05), duplicate_of="0001")
    for r in range(1, 7):
        history.append_round(
            paths,
            RoundRecord(
                r,
                (r,),
                BASE_SHA,
                (r,) if r != 4 else (),
                (r,) if r in (1, 3, 6) else (),
                1.05,
                100.0 + r,
                200.0,
                Usage(),
                ("boom",) if r == 4 else (),
                "t",
            ),
        )
    st = compute_status(
        paths, rounds_total=32, model="deepseek/deepseek-v4-pro-0813", run_id="t1_w1"
    )
    assert st.rounds_done == 6 and st.rounds_total == 32 and st.attempts == 6
    assert st.measured == 5 and st.no_patch == 1 and st.duplicates == 1
    assert st.tests_pass == 4 and st.clears_noise == 3
    assert st.best_ratio == 1.05 and st.best_attempt == 1
    assert st.best_raw_ratio == 1.30
    assert st.by_stop == {"submitted": 5, "max_turns": 1}
    assert st.worker_wall_median_s == 103.5 and st.referee_wall_median_s == 200.0
    assert st.usage == Usage(6000, 3600, 600, 240)
    assert st.prompt_tokens_per_attempt == 1000 and st.cached_share == 0.6
    assert st.cost_usd is not None and abs(st.cost_usd - (6000 * 1.32 + 600 * 3.96) / 1e6) < 1e-9
    assert st.harness_errors == ("boom",)
    text = st.render()
    assert "6 of 32 rounds" in text
    assert "real speedups 3" in text
    assert "best real speedup so far 1.0500 (attempt 0001)" in text
    assert "best raw ratio over all timed attempts 1.3000" in text
    assert "boom" in text


def test_status_of_an_empty_run(tmp_path: Path) -> None:
    st = compute_status(history.RunPaths(tmp_path), 32, "unknown-model", "x")
    assert st.attempts == 0 and st.best_ratio is None and st.cost_usd is None
    text = st.render()
    assert "best real speedup so far: none" in text and "harness errors: none" in text
