from pathlib import Path

from autoresearch import history
from autoresearch.orchestrator.status import compute_status
from autoresearch.types import AttemptRef, RefereeResult, RoundRecord, Usage, Verdict
from tests.helpers import diff_for, submitted


def _attempt(paths: history.RunPaths, n: int, verdict: Verdict, ratio: float | None) -> None:
    ref = AttemptRef(n, n, 0)
    out = submitted(diff_for(str(n)) if verdict != Verdict.NO_PATCH else None)
    history.write_attempt(paths, ref, "sha", {}, out, "")
    history.write_result(paths, ref, RefereeResult(verdict, "r", 1.0106, median_ratio=ratio))


def test_status_totals(tmp_path: Path) -> None:
    paths = history.RunPaths(tmp_path)
    _attempt(paths, 1, Verdict.ACCEPTED, 1.05)
    _attempt(paths, 2, Verdict.REJECTED_BELOW_THRESHOLD, 1.002)
    _attempt(paths, 3, Verdict.ACCEPTED, 1.02)
    _attempt(paths, 4, Verdict.NO_PATCH, None)
    for r in (1, 2, 3, 4):
        history.append_round(
            paths,
            RoundRecord(
                r,
                (r,),
                "a",
                "b",
                (r,) if r in (1, 3) else (),
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
    assert st.rounds_done == 4 and st.rounds_total == 32 and st.attempts == 4
    assert st.by_verdict == {"accepted": 2, "rejected_below_threshold": 1, "no_patch": 1}
    assert st.by_stop == {"submitted": 3, "max_turns": 1}
    assert st.accepted == 2 and st.best_ratio == 1.05
    assert abs(st.cumulative_ratio - 1.05 * 1.02) < 1e-9
    assert st.worker_wall_median_s == 102.5 and st.referee_wall_median_s == 200.0
    assert st.usage == Usage(4000, 2400, 400, 160)
    assert st.prompt_tokens_per_attempt == 1000 and st.cached_share == 0.6
    assert st.cost_usd is not None and abs(st.cost_usd - (4000 * 1.32 + 400 * 3.96) / 1e6) < 1e-9
    assert st.harness_errors == ("boom",)
    text = st.render()
    assert "4 of 32 rounds" in text and "accepted 2" in text and "boom" in text


def test_status_of_an_empty_run(tmp_path: Path) -> None:
    st = compute_status(history.RunPaths(tmp_path), 32, "unknown-model", "x")
    assert st.attempts == 0 and st.best_ratio is None and st.cost_usd is None
    assert "accepted 0" in st.render() and "harness errors: none" in st.render()
