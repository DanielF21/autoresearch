"""One round of the orchestrator with a fake worker, fake referee boxes, and a
real local git incumbent. The fake boxes answer the incumbent sync with the
tree hash a real box would produce, so the orchestrator's own checks run."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch import history, incumbent
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig, load_config
from autoresearch.orchestrator import run as run_mod
from autoresearch.orchestrator.pool import RefereePool
from autoresearch.orchestrator.round import run_round
from autoresearch.types import Verdict
from tests.helpers import FakeWorker, diff_for, referee_box, submitted, tree_oracle

ROOT = Path(__file__).parent.parent

OTHER_FILE_DIFF = (
    "diff --git a/networkx/algorithms/other.py b/networkx/algorithms/other.py\n"
    "--- a/networkx/algorithms/other.py\n"
    "+++ b/networkx/algorithms/other.py\n"
    "@@ -1 +1 @@\n"
    "-y = 1\n"
    "+y = 2\n"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def origin(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "origin"
    (repo / "networkx" / "algorithms").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "networkx" / "algorithms" / "cluster.py").write_text("x = 1\n")
    (repo / "networkx" / "algorithms" / "other.py").write_text("y = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pinned")
    return repo, _git(repo, "rev-parse", "HEAD")


def make_config(origin: tuple[Path, str], width: int) -> RunConfig:
    cfg = load_config(ROOT / "configs" / "t1_w1.toml")
    target = replace(cfg.target, repo=str(origin[0]), sha=origin[1], docs=())
    return replace(cfg, width=width, rounds=3, target=target)


def make_factory(
    origin: tuple[Path, str], speedup_of: dict[str, float], *, first_box_broken: bool = False
) -> FakeBoxFactory:
    def prepare(box: FakeBox, role: str) -> None:
        assert role == "referee"
        referee_box(
            box,
            speedup=lambda patch: next((v for k, v in speedup_of.items() if k in patch), 1.0),
            tree_oracle=tree_oracle(box, origin[0], origin[1]),
        )
        if first_box_broken and box.box_id == "sb_fake_1":

            def explode(_cmd: str) -> object:
                raise BoxError("host lost")

            box.on("apply_patch.py", explode, first=True)  # type: ignore[arg-type]

    return FakeBoxFactory(prepare=prepare)


def start(
    tmp_path: Path, origin: tuple[Path, str], width: int, speedup_of: dict[str, float], **kw: bool
) -> tuple[RunConfig, history.RunPaths, RefereePool, FakeBoxFactory]:
    cfg = make_config(origin, width)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    factory = make_factory(origin, speedup_of, **kw)
    pool = RefereePool(cfg, paths, factory)
    pool.start()
    inc = paths.incumbent
    pool.sync_all(
        cfg.target.sha, incumbent.cumulative_diff(inc, cfg.target.sha), incumbent.tree_hash(inc)
    )
    return cfg, paths, pool, factory


def test_width_one_accepts_and_advances_the_incumbent(
    tmp_path: Path, origin: tuple[Path, str]
) -> None:
    cfg, paths, pool, factory = start(tmp_path, origin, 1, {"x = fast": 1.05})
    worker = FakeWorker([submitted(diff_for("fast"))])
    outcome = run_round(1, cfg, paths, worker, pool)

    assert outcome.results[0].verdict == Verdict.ACCEPTED
    rec = outcome.record
    assert rec.round == 1 and rec.attempt_numbers == (1,) and rec.accepted_numbers == (1,)
    assert rec.incumbent_sha_before == origin[1] and rec.incumbent_sha_after != origin[1]
    assert rec.errors == ()
    assert (paths.incumbent / "networkx" / "algorithms" / "cluster.py").read_text() == "x = fast\n"
    assert incumbent.log_shas(paths.incumbent, origin[1]) == (rec.incumbent_sha_after,)

    (attempt,) = history.load_history(paths)
    assert attempt.ref.number == 1 and attempt.patch == diff_for("fast")
    assert attempt.result is not None and attempt.result.verdict == Verdict.ACCEPTED
    assert history.read_rounds(paths)[0] == rec

    inp = worker.inputs[0]
    assert inp.incumbent_sha == origin[1] and inp.history == () and inp.stack_diff == ""
    assert inp.cache_key.startswith("t1_w1-")

    # The referee was synced to the new incumbent for the next round.
    box = factory.created[0]
    assert box.files["/workspace/work/incumbent.diff"].decode() == incumbent.cumulative_diff(
        paths.incumbent, origin[1]
    )


def test_second_round_sees_history_and_the_stack(tmp_path: Path, origin: tuple[Path, str]) -> None:
    cfg, paths, pool, _ = start(tmp_path, origin, 1, {"x = fast": 1.05})
    worker = FakeWorker([submitted(diff_for("fast")), submitted(None)])
    run_round(1, cfg, paths, worker, pool)
    outcome = run_round(2, cfg, paths, worker, pool)
    assert outcome.results[0].verdict == Verdict.NO_PATCH
    inp = worker.inputs[1]
    assert [a.ref.number for a in inp.history] == [1]
    assert "+x = fast" in inp.stack_diff
    assert inp.incumbent_sha == history.read_rounds(paths)[0].incumbent_sha_after
    assert history.next_attempt_number(paths) == 3


def test_duplicate_of_history_is_not_judged(tmp_path: Path, origin: tuple[Path, str]) -> None:
    cfg, paths, pool, factory = start(tmp_path, origin, 1, {"x = slow": 1.0})
    worker = FakeWorker([submitted(diff_for("slow")), submitted(diff_for("slow"))])
    run_round(1, cfg, paths, worker, pool)
    judged_before = len([c for c in factory.created[0].commands if "run_tests.py" in c])
    outcome = run_round(2, cfg, paths, worker, pool)
    assert outcome.results[0].verdict == Verdict.DUPLICATE
    assert "0001" in outcome.results[0].reason
    judged_after = len([c for c in factory.created[0].commands if "run_tests.py" in c])
    assert judged_after == judged_before


def test_width_two_conflicting_patches_second_is_superseded(
    tmp_path: Path, origin: tuple[Path, str]
) -> None:
    cfg, paths, pool, _ = start(tmp_path, origin, 2, {"x = best": 1.10, "x = good": 1.05})
    worker = FakeWorker([submitted(diff_for("good")), submitted(diff_for("best"))])
    outcome = run_round(1, cfg, paths, worker, pool)
    assert outcome.results[1].verdict == Verdict.ACCEPTED
    assert outcome.results[0].verdict == Verdict.SUPERSEDED
    assert "no longer applies" in outcome.results[0].reason
    assert outcome.record.accepted_numbers == (2,)
    assert (paths.incumbent / "networkx" / "algorithms" / "cluster.py").read_text() == "x = best\n"


def test_width_two_compatible_patches_both_merge_after_rejudge(
    tmp_path: Path, origin: tuple[Path, str]
) -> None:
    cfg, paths, pool, factory = start(tmp_path, origin, 2, {"x = best": 1.10, "y = 2": 1.05})
    worker = FakeWorker([submitted(OTHER_FILE_DIFF), submitted(diff_for("best"))])
    outcome = run_round(1, cfg, paths, worker, pool)
    assert outcome.results[1].verdict == Verdict.ACCEPTED
    assert outcome.results[0].verdict == Verdict.ACCEPTED
    assert outcome.record.accepted_numbers == (2, 1)
    assert len(incumbent.log_shas(paths.incumbent, origin[1])) == 2
    # The second patch was judged twice: alone, then on top of the first.
    judged = [c for c in factory.created[0].commands if "--scope full" in c]
    assert len(judged) == 2


def test_referee_box_failure_is_rebuilt_and_retried(
    tmp_path: Path, origin: tuple[Path, str]
) -> None:
    cfg, paths, pool, factory = start(
        tmp_path, origin, 1, {"x = fast": 1.05}, first_box_broken=True
    )
    worker = FakeWorker([submitted(diff_for("fast"))])
    outcome = run_round(1, cfg, paths, worker, pool)
    assert outcome.results[0].verdict == Verdict.ACCEPTED
    assert len(factory.created) == 2
    assert factory.created[0].terminated and not factory.created[1].terminated
    assert history.read_boxes(paths)["referee:0"] == "sb_fake_2"
    assert outcome.record.errors == ()


def test_run_loop_resumes_after_the_last_completed_round(
    tmp_path: Path, origin: tuple[Path, str]
) -> None:
    cfg = make_config(origin, 1)
    paths = run_mod.init_run(cfg, tmp_path / "run", (("profile.txt", "hot"),))
    assert (paths.target / "profile.txt").read_text() == "hot"
    factory = make_factory(origin, {"x = fast": 1.05})
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


def test_half_written_round_refuses_to_resume(tmp_path: Path, origin: tuple[Path, str]) -> None:
    cfg = make_config(origin, 1)
    paths = run_mod.init_run(cfg, tmp_path / "run", ())
    from autoresearch.types import AttemptRef

    history.write_attempt(paths, AttemptRef(1, 1, 0), origin[1], {}, submitted(None), "")
    with pytest.raises(run_mod.RunError, match="move the extra attempt"):
        run_mod.run(cfg, paths, FakeWorker([submitted(None)]), make_factory(origin, {}))
    assert not paths.lock.exists()


def test_init_run_refuses_a_changed_config(tmp_path: Path, origin: tuple[Path, str]) -> None:
    cfg = make_config(origin, 1)
    run_mod.init_run(cfg, tmp_path / "run", ())
    other = replace(cfg, source_text=cfg.source_text + "\n# changed\n")
    with pytest.raises(run_mod.RunError, match="never changes"):
        run_mod.init_run(other, tmp_path / "run", ())


def test_rebuild_incumbent_replays_accepted_patches(
    tmp_path: Path, origin: tuple[Path, str]
) -> None:
    cfg, paths, pool, _ = start(tmp_path, origin, 1, {"x = fast": 1.05})
    run_round(1, cfg, paths, FakeWorker([submitted(diff_for("fast"))]), pool)
    tree_before = incumbent.tree_hash(paths.incumbent)
    import shutil

    shutil.rmtree(paths.incumbent)
    from autoresearch.orchestrator.round import rebuild_incumbent

    rebuild_incumbent(cfg, paths)
    assert (paths.incumbent / "networkx" / "algorithms" / "cluster.py").read_text() == "x = fast\n"
    assert incumbent.tree_hash(paths.incumbent) == tree_before
    assert len(incumbent.log_shas(paths.incumbent, origin[1])) == 1
