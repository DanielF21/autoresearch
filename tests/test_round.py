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
from autoresearch.types import AttemptRef, WorkerOutput
from autoresearch.worker.protocol import WorkerInput
from tests.helpers import BASE_SHA, FakeWorker, diff_for, referee_box, submitted

ROOT = Path(__file__).parent.parent


def make_config(width: int) -> RunConfig:
    cfg = load_config(ROOT / "configs" / "t1_w4d.toml")
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
    assert m.clears_noise and m.speedup == pytest.approx(1.05)
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
    assert inp.cache_key.startswith("t1_w4d-")
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
    assert outcome.measurements[0].speedup == pytest.approx(1.05)
    assert outcome.measurements[1].speedup == pytest.approx(1.10)
    assert outcome.record.clears_noise_numbers == (1, 2)
    assert outcome.record.best_ratio_so_far == pytest.approx(1.10)
    # Each slot measured exactly once, on its own referee box.
    for box in factory.created:
        assert len([c for c in box.commands if "--scope full" in c]) == 1


def test_failed_and_slow_patches_are_measured_and_kept(tmp_path: Path) -> None:
    cfg, paths, pool, _ = start(tmp_path, 1, {"x = slow": 0.8})
    outcome = run_round(1, cfg, paths, FakeWorker([submitted(diff_for("slow"))]), pool)
    m = outcome.measurements[0]
    assert m.speedup == pytest.approx(0.8) and not m.clears_noise
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
    # Stopping early keeps nothing: a box left for a resume bills until one comes.
    assert [b.terminated for b in factory.created] == [True]
    assert history.read_boxes(paths) == {}

    assert run_mod.run(cfg, paths, worker, factory) == 3
    assert [r.round for r in history.read_rounds(paths)] == [1, 2, 3]
    assert len(factory.created) == 2  # the resume built its own referee
    assert all(b.terminated for b in factory.created)
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=paths.root, capture_output=True, text=True
    ).stdout
    assert "round 3" in log and "run initialised" in log
    assert not (paths.root / "incumbent").exists()


class RaisingWorker:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def attempt(self, inp: WorkerInput) -> WorkerOutput:
        raise self.error


@pytest.mark.parametrize("error", [RuntimeError("worker bug"), SystemExit(143)])
def test_run_terminates_referees_when_a_round_raises(tmp_path: Path, error: BaseException) -> None:
    """SystemExit is what ``shutdown`` turns SIGTERM and SIGHUP into."""
    cfg = make_config(2)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    factory = make_factory({})
    with pytest.raises(type(error)):
        run_mod.run(cfg, paths, RaisingWorker(error), factory)
    assert len(factory.created) == 2 and all(b.terminated for b in factory.created)
    assert history.read_boxes(paths) == {}
    assert not paths.lock.exists()


def test_a_referee_that_fails_setup_is_terminated(tmp_path: Path) -> None:
    cfg = make_config(1)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())

    def prepare(box: FakeBox, role: str) -> None:
        def explode(_cmd: str) -> object:
            raise BoxError("setup lost the host")

        box.on("", explode)  # type: ignore[arg-type]

    factory = FakeBoxFactory(prepare=prepare)
    with pytest.raises(BoxError, match="setup lost the host"):
        run_mod.run(cfg, paths, FakeWorker([]), factory)
    assert len(factory.created) == 1 and factory.created[0].terminated
    assert history.read_boxes(paths) == {}
    assert not paths.lock.exists()


def test_the_next_launch_terminates_what_a_killed_orchestrator_left(tmp_path: Path) -> None:
    cfg = make_config(1)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    factory = make_factory({})
    orphan = factory.create(name="referee-killed-0", role="referee")
    assert isinstance(orphan, FakeBox)
    history.write_boxes(paths, {"referee:0": orphan.box_id})

    pool = RefereePool(cfg, paths, factory)
    pool.start()
    assert orphan.terminated
    assert history.read_boxes(paths) == {"referee:0": factory.created[1].box_id}
    assert pool.terminate_all() == ()
    assert history.read_boxes(paths) == {}


def test_a_finished_run_still_terminates_what_a_killed_launch_left(tmp_path: Path) -> None:
    """The early return for a complete run never starts the pool, and must not skip this."""
    cfg = make_config(1)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    factory = make_factory({})
    worker = FakeWorker([submitted(None), submitted(None), submitted(None)])
    assert run_mod.run(cfg, paths, worker, factory) == 3
    orphan = factory.create(name="referee-killed-0", role="referee")
    assert isinstance(orphan, FakeBox)
    history.write_boxes(paths, {"referee:0": orphan.box_id})
    assert run_mod.run(cfg, paths, worker, factory) == 3
    assert orphan.terminated and history.read_boxes(paths) == {}


def test_a_referee_that_will_not_terminate_is_logged_kept_and_raised(tmp_path: Path) -> None:
    cfg = make_config(1)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    factory = make_factory({})
    referee = factory.prepare
    assert referee is not None

    def prepare(box: FakeBox, role: str) -> None:
        referee(box, role)

        def stuck() -> None:
            raise BoxError("terminate did not return within 300s")

        box.terminate = stuck  # type: ignore[method-assign]

    factory.prepare = prepare
    log = tmp_path / "run.log"
    with pytest.raises(BoxError, match="may still be running"):
        run_mod.run(cfg, paths, FakeWorker([submitted(None)]), factory, until_round=1, log=log)
    assert list(history.read_boxes(paths).values()) == [factory.created[0].box_id]
    assert "referee box may still be running" in log.read_text()
    assert history.last_completed_round(paths) == 1
    assert not paths.lock.exists()


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
