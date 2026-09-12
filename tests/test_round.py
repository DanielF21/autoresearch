"""One round of the orchestrator with a fake worker and fake referee boxes.

Nothing is merged and nothing needs a git repository on the host: every
attempt starts from and is measured against the same base commit.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch import history
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig, load_config
from autoresearch.orchestrator import run as run_mod
from autoresearch.orchestrator.pool import RefereePool
from autoresearch.orchestrator.round import NO_PATCH, run_round
from autoresearch.types import AttemptRef
from tests.helpers import BASE_SHA, FakeWorker, diff_for, referee_box, submitted

ROOT = Path(__file__).parent.parent


def make_config(width: int) -> RunConfig:
    cfg = load_config(ROOT / "configs" / "t1_w1.toml")
    return replace(cfg, width=width, rounds=3, target=replace(cfg.target, docs=()))


def make_factory(speedup_of: dict[str, float], *, first_box_broken: bool = False) -> FakeBoxFactory:
    def prepare(box: FakeBox, role: str) -> None:
        assert role == "referee"
        referee_box(
            box,
            speedup=lambda patch: next((v for k, v in speedup_of.items() if k in patch), 1.0),
        )
        if first_box_broken and box.box_id == "sb_fake_1":

            def explode(_cmd: str) -> object:
                raise BoxError("host lost")

            box.on("apply_patch.py", explode, first=True)  # type: ignore[arg-type]

    return FakeBoxFactory(prepare=prepare)


def start(
    tmp_path: Path, width: int, speedup_of: dict[str, float], **kw: bool
) -> tuple[RunConfig, history.RunPaths, RefereePool, FakeBoxFactory]:
    cfg = make_config(width)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    factory = make_factory(speedup_of, **kw)
    pool = RefereePool(cfg, paths, factory)
    pool.start()
    return cfg, paths, pool, factory


def test_width_one_measures_and_records_best_so_far(tmp_path: Path) -> None:
    cfg, paths, pool, factory = start(tmp_path, 1, {"x = fast": 1.05})
    worker = FakeWorker([submitted(diff_for("fast"))])
    outcome = run_round(1, cfg, paths, worker, pool)

    m = outcome.measurements[0]
    assert m.clears_noise and m.median_ratio == pytest.approx(1.05)
    rec = outcome.record
    assert rec.round == 1 and rec.attempt_numbers == (1,)
    assert rec.base_sha == BASE_SHA
    assert rec.measured_numbers == (1,) and rec.clears_noise_numbers == (1,)
    assert rec.best_ratio_so_far == pytest.approx(1.05)
    assert rec.errors == ()

    (attempt,) = history.load_history(paths)
    assert attempt.ref.number == 1 and attempt.patch == diff_for("fast")
    assert attempt.base_sha == BASE_SHA and attempt.skipped == "" and attempt.duplicate_of == ""
    assert attempt.measurement == m
    assert history.read_rounds(paths)[0] == rec

    inp = worker.inputs[0]
    assert inp.base_sha == BASE_SHA and inp.history == ()
    assert inp.cache_key.startswith("t1_w1-")
    # The referee box was never asked to move off the base.
    assert not any("git apply" in c for c in factory.created[0].commands)


def test_second_round_sees_history_and_starts_from_the_same_base(tmp_path: Path) -> None:
    cfg, paths, pool, _ = start(tmp_path, 1, {"x = fast": 1.05})
    worker = FakeWorker([submitted(diff_for("fast")), submitted(None)])
    run_round(1, cfg, paths, worker, pool)
    outcome = run_round(2, cfg, paths, worker, pool)
    assert outcome.measurements == {}
    assert outcome.record.measured_numbers == ()
    assert outcome.record.best_ratio_so_far == pytest.approx(1.05)  # carried from round 1
    inp = worker.inputs[1]
    assert [a.ref.number for a in inp.history] == [1]
    assert inp.base_sha == BASE_SHA
    second = history.load_history(paths)[1]
    assert second.skipped == NO_PATCH and second.measurement is None
    assert history.next_attempt_number(paths) == 3


def test_duplicate_is_recorded_and_measured_again(tmp_path: Path) -> None:
    cfg, paths, pool, factory = start(tmp_path, 1, {"x = slow": 1.0})
    worker = FakeWorker([submitted(diff_for("slow")), submitted(diff_for("slow"))])
    run_round(1, cfg, paths, worker, pool)
    outcome = run_round(2, cfg, paths, worker, pool)
    assert 0 in outcome.measurements
    attempts = history.load_history(paths)
    assert attempts[0].duplicate_of == "" and attempts[1].duplicate_of == "0001"
    assert attempts[1].measurement is not None
    full_runs = [c for c in factory.created[0].commands if "--scope full" in c]
    assert len(full_runs) == 2


def test_width_two_records_both_and_best_is_the_maximum(tmp_path: Path) -> None:
    cfg, paths, pool, factory = start(tmp_path, 2, {"x = best": 1.10, "x = good": 1.05})
    worker = FakeWorker([submitted(diff_for("good")), submitted(diff_for("best"))])
    outcome = run_round(1, cfg, paths, worker, pool)
    assert outcome.measurements[0].median_ratio == pytest.approx(1.05)
    assert outcome.measurements[1].median_ratio == pytest.approx(1.10)
    assert outcome.record.clears_noise_numbers == (1, 2)
    assert outcome.record.best_ratio_so_far == pytest.approx(1.10)
    # Each slot measured exactly once, on its own referee box.
    for box in factory.created:
        assert len([c for c in box.commands if "--scope full" in c]) == 1


def test_failed_and_slow_patches_are_measured_and_kept(tmp_path: Path) -> None:
    cfg, paths, pool, _ = start(tmp_path, 1, {"x = slow": 0.8})
    outcome = run_round(1, cfg, paths, FakeWorker([submitted(diff_for("slow"))]), pool)
    m = outcome.measurements[0]
    assert m.median_ratio == pytest.approx(0.8) and not m.clears_noise
    assert outcome.record.measured_numbers == (1,) and outcome.record.clears_noise_numbers == ()
    assert outcome.record.best_ratio_so_far is None
    assert history.load_history(paths)[0].measurement == m


def test_referee_box_failure_is_rebuilt_and_retried(tmp_path: Path) -> None:
    cfg, paths, pool, factory = start(tmp_path, 1, {"x = fast": 1.05}, first_box_broken=True)
    outcome = run_round(1, cfg, paths, FakeWorker([submitted(diff_for("fast"))]), pool)
    assert outcome.measurements[0].clears_noise
    assert len(factory.created) == 2
    assert factory.created[0].terminated and not factory.created[1].terminated
    assert history.read_boxes(paths)["referee:0"] == "sb_fake_2"
    assert outcome.record.errors == ()


def test_run_loop_resumes_after_the_last_completed_round(tmp_path: Path) -> None:
    cfg = make_config(1)
    paths = run_mod.init_run(cfg, tmp_path / "run", (("profile.txt", "hot"),))
    assert (paths.target / "profile.txt").read_text() == "hot"
    factory = make_factory({"x = fast": 1.05})
    worker = FakeWorker([submitted(diff_for("fast")), submitted(None), submitted(None)])

    assert run_mod.run(cfg, paths, worker, factory, until_round=1) == 1
    assert history.last_completed_round(paths) == 1
    assert not paths.lock.exists()
    assert factory.created[0].terminated is False  # referees persist between calls

    assert run_mod.run(cfg, paths, worker, factory) == 3
    assert [r.round for r in history.read_rounds(paths)] == [1, 2, 3]
    assert all(b.terminated for b in factory.created)  # run complete: boxes released
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=paths.root, capture_output=True, text=True
    ).stdout
    assert "round 3" in log and "run initialised" in log
    assert not (paths.root / "incumbent").exists()


def test_half_written_round_refuses_to_resume(tmp_path: Path) -> None:
    cfg = make_config(1)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    history.write_attempt(paths, AttemptRef(1, 1, 0), BASE_SHA, {}, submitted(None), "")
    with pytest.raises(run_mod.RunError, match="move the extra attempt"):
        run_mod.run(cfg, paths, FakeWorker([submitted(None)]), make_factory({}))
    assert not paths.lock.exists()


def test_init_run_refuses_a_changed_config(tmp_path: Path) -> None:
    cfg = make_config(1)
    run_mod.init_run(cfg, tmp_path / "run", ())
    other = replace(cfg, source_text=cfg.source_text + "\n# changed\n")
    with pytest.raises(run_mod.RunError, match="never changes"):
        run_mod.init_run(other, tmp_path / "run", ())
