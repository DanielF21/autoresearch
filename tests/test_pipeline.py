"""The stage read from disk, and auto driving every stage in order, against fakes.

Most tests drive ``pipeline.auto`` with steps that write what the real commands
write, so the order, resumption, repair and stopping rules are tested without the
commands. Two tests go through ``main`` with the real commands over a fake model and
fake boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autoresearch import env, history, pipeline
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory
from autoresearch.cli import main
from autoresearch.config import load_config
from autoresearch.config_edit import set_docs
from autoresearch.intake import derive
from autoresearch.model.fake_model import FakeChatModel, tool_call
from autoresearch.pipeline import Stage
from autoresearch.referee import admissibility
from autoresearch.types import RoundRecord, Usage
from tests.helpers import referee_box
from tests.test_cli_intake import GN800_FLOOR, PILOT
from tests.test_intake import GOOD, _git_repo

URL = "https://github.com/org/widget"
RUN_ID = "t1_w4d"


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "load_dotenv", lambda *a, **k: None)


def config_text(seed: int, *, docs: bool, floor: bool) -> str:
    text = PILOT.read_text().replace("rounds = 32", "rounds = 2", 1)
    if not floor:
        text = text.replace(GN800_FLOOR, "", 1)
    text = set_docs(text, ["doc.txt"] if docs else [])
    return text.replace(
        "nx.gn_graph(800, seed=3 + SEED)", f"nx.gn_graph(800, seed={seed} + SEED)", 1
    )


@dataclass
class World:
    """Steps that write what each real command writes, and record that they ran."""

    ctx: pipeline.Context
    checks: list[bool] = field(default_factory=lambda: [True])
    calls: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    run_rc: int = 0
    repair_rc: int = 0
    calibrate_all: bool = True
    seed: int = 3

    def steps(self) -> pipeline.Steps:
        return pipeline.Steps(
            intake=self.intake,
            propose=self.propose,
            repair=self.repair,
            check=self.check,
            profile=self.profile,
            calibrate=self.calibrate,
            run=self.run,
            scribe=self.scribe,
        )

    def _config(self, *, docs: bool, floor: bool) -> None:
        self.ctx.config_path().parent.mkdir(parents=True, exist_ok=True)
        self.ctx.config_path().write_text(config_text(self.seed, docs=docs, floor=floor))

    def intake(self, ctx: pipeline.Context) -> int:
        self.calls.append("intake")
        cfg = load_config(PILOT)
        draft = derive.Draft(
            name="widget",
            repo=URL,
            sha=cfg.target.sha,
            package="networkx",
            package_root=".",
            pip=(),
            tests_full="networkx",
            allow=(),
            deny=(),
            run_id=RUN_ID,
            template=str(PILOT),
        )
        ctx.draft_dir.mkdir(parents=True)
        (ctx.draft_dir / derive.DRAFT_FILE).write_text(json.dumps(draft.to_dict()))
        return 0

    def propose(self, ctx: pipeline.Context) -> int:
        self.calls.append("propose")
        self._config(docs=False, floor=False)
        return 0

    def repair(self, ctx: pipeline.Context, report: str) -> int:
        self.calls.append("repair")
        self.reports.append(report)
        if self.repair_rc:
            return self.repair_rc
        self.seed += 1
        self._config(docs=False, floor=False)
        return 0

    def check(self, ctx: pipeline.Context) -> int:
        self.calls.append("check")
        passed = self.checks.pop(0)
        cfg = load_config(ctx.config_path())
        level = "pass" if passed else "fail"
        record = {
            "sha": cfg.target.sha,
            "target_hash": admissibility.admission_hash(cfg.target),
            "verdicts": [{"rule": "gn800 call length", "level": level, "detail": "too long"}],
            "report": f"{level.upper():<5} gn800 call length  too long",
        }
        (ctx.runs_root / "check").mkdir(parents=True, exist_ok=True)
        (ctx.runs_root / "check" / f"{len(self.calls):04d}.json").write_text(json.dumps(record))
        return 0 if passed else 1

    def profile(self, ctx: pipeline.Context) -> int:
        self.calls.append("profile")
        (ctx.repo_root / "doc.txt").write_text("profile\n")
        self._config(docs=True, floor=False)
        return 0

    def calibrate(self, ctx: pipeline.Context) -> int:
        self.calls.append("calibrate")
        if not self.calibrate_all:
            return 1
        self._config(docs=True, floor=True)
        return 0

    def run(self, ctx: pipeline.Context) -> int:
        self.calls.append("run")
        if self.run_rc:
            return self.run_rc
        paths = history.RunPaths(ctx.run_dir())
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.config.write_text(ctx.config_path().read_text())
        for n in (1, 2):
            history.append_round(
                paths,
                RoundRecord(n, (), "sha", (), (), None, 1.0, 1.0, Usage(), (), "now"),
            )
        return 0

    def scribe(self, ctx: pipeline.Context) -> int:
        self.calls.append("scribe")
        out = ctx.scribe_root / RUN_ID / "20260913T000000Z"
        out.mkdir(parents=True)
        (out / pipeline.PICK_FILE).write_text(json.dumps({"attempt": 3, "reasons": "small"}))
        (out / "pr.md").write_text("# t\n\nb\n")
        (out / "patch.diff").write_text("diff\n")
        return 0


def _ctx(tmp_path: Path) -> pipeline.Context:
    return pipeline.Context(
        url=URL,
        width=4,
        rounds=2,
        template=PILOT,
        intake_root=tmp_path / "auto",
        configs=tmp_path / "configs",
        runs_root=tmp_path / "runs",
        scribe_root=tmp_path / "runs" / "scribe",
        docs_root=tmp_path / "docs",
        calibration_out=tmp_path / "cal",
        repo_root=tmp_path,
    )


EVERY_STAGE = ["intake", "propose", "check", "profile", "calibrate", "run", "scribe"]


def test_without_yes_only_the_free_intake_runs_and_the_plan_is_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(_ctx(tmp_path))
    assert pipeline.auto(world.ctx, world.steps(), yes=False) == 0
    assert pipeline.auto(world.ctx, world.steps(), yes=False) == 0
    assert world.calls == ["intake"]
    out = capsys.readouterr().out
    assert "width 4, 2 rounds" in out and "8 model attempts" in out
    assert "4 worker boxes" in out and "never measured on this target" in out
    assert "nothing spent" in out


def test_yes_runs_every_stage_in_order_and_a_second_invocation_does_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(_ctx(tmp_path))
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 0
    assert world.calls == EVERY_STAGE
    out = capsys.readouterr().out
    assert "done: t1_w4d" in out and "pr.md" in out and "patch.diff" in out

    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 0
    assert world.calls == EVERY_STAGE
    assert "== stage: done" in capsys.readouterr().out


def test_a_failed_run_stops_and_the_next_invocation_resumes_at_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(_ctx(tmp_path), run_rc=1)
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 1
    assert world.calls == EVERY_STAGE[:-1]
    assert "ready to run did not advance" in capsys.readouterr().out

    world.run_rc = 0
    world.calls.clear()
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 0
    assert world.calls == ["run", "scribe"]


def test_a_failed_check_goes_back_to_the_proposal_with_its_report(tmp_path: Path) -> None:
    world = World(_ctx(tmp_path), checks=[False, True])
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 0
    assert world.calls == ["intake", "propose", "check", "repair", "check", *EVERY_STAGE[3:]]
    assert world.reports == ["FAIL  gn800 call length  too long"]
    assert (world.ctx.draft_dir / "rejected" / "1.toml").is_file()
    (repair,) = json.loads((world.ctx.draft_dir / pipeline.REPAIRS_FILE).read_text())
    assert repair["failed"] == ["gn800 call length"]


def test_repairs_stop_after_the_limit_and_nothing_is_profiled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(_ctx(tmp_path), checks=[False] * (pipeline.MAX_REPAIRS + 1))
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 1
    assert world.calls.count("repair") == pipeline.MAX_REPAIRS
    assert world.calls.count("check") == pipeline.MAX_REPAIRS + 1
    assert "profile" not in world.calls
    out = capsys.readouterr().out
    assert f"after {pipeline.MAX_REPAIRS} repairs: gn800 call length" in out


def test_a_repair_that_fails_puts_the_checked_config_back(tmp_path: Path) -> None:
    world = World(_ctx(tmp_path), checks=[False], repair_rc=1)
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 1
    assert world.ctx.config_path().is_file()
    assert not (world.ctx.draft_dir / "rejected" / "1.toml").exists()
    assert pipeline.where(world.ctx).stage is Stage.CHECK_FAILED
    assert world.calls.count("propose") == 1


def test_a_calibration_that_leaves_an_input_without_a_floor_stops_before_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(_ctx(tmp_path), calibrate_all=False)
    assert pipeline.auto(world.ctx, world.steps(), yes=True) == 1
    assert "run" not in world.calls
    assert "still no floor for gn800" in capsys.readouterr().out


def test_until_stops_once_the_stage_is_reached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(_ctx(tmp_path))
    assert pipeline.auto(world.ctx, world.steps(), yes=True, until="calibrated") == 0
    assert world.calls == EVERY_STAGE[:5]
    assert "as --until calibrated asked" in capsys.readouterr().out
    assert pipeline.auto(world.ctx, world.steps(), yes=True, until="bogus") == 2


def test_run_stage_reads_rounds_and_the_scribe_pick(tmp_path: Path) -> None:
    world = World(_ctx(tmp_path))
    assert pipeline.auto(world.ctx, world.steps(), yes=True, until="calibrated") == 0
    cfg = load_config(world.ctx.config_path())
    runs, scribe = world.ctx.runs_root, world.ctx.scribe_root
    assert pipeline.run_stage(cfg, runs, scribe).stage is Stage.READY
    paths = history.RunPaths(world.ctx.run_dir())
    paths.root.mkdir(parents=True)
    paths.config.write_text(cfg.source_text)
    history.append_round(paths, RoundRecord(1, (), "s", (), (), None, 1.0, 1.0, Usage(), (), "t"))
    assert pipeline.run_stage(cfg, runs, scribe) == pipeline.Where(
        Stage.RUN_INCOMPLETE, rounds_done=1
    )
    world.run(world.ctx)
    assert pipeline.run_stage(cfg, runs, scribe).stage is Stage.NEEDS_SCRIBE
    world.scribe(world.ctx)
    assert pipeline.run_stage(cfg, runs, scribe).stage is Stage.DONE


# ----- through the command line ------------------------------------------------------


def test_auto_needs_width_and_rounds(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["auto", URL])
    assert "--width" in capsys.readouterr().err


def _no_spend(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch.boxes import sail_box
    from autoresearch.model import sail_model

    def refuse(*a: object, **k: object) -> None:
        raise AssertionError("nothing may be created without --yes")

    monkeypatch.setattr(sail_model, "SailChatModel", refuse)
    monkeypatch.setattr(sail_box, "SailBoxFactory", refuse)


def test_auto_without_yes_takes_the_repository_in_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    _no_spend(monkeypatch)
    args = ["auto", f"file://{src}", "--width", "4", "--rounds", "2", "--template", str(PILOT)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "nothing spent" in out and "8 model attempts" in out
    assert (tmp_path / "runs" / "auto" / "repo" / derive.DRAFT_FILE).is_file()
    assert not (tmp_path / "configs").exists()


def test_auto_takes_a_repository_to_a_calibrated_config_with_the_real_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoresearch.boxes import sail_box
    from autoresearch.model import sail_model

    src = _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    draft_file = tmp_path / "runs" / "auto" / "repo" / derive.DRAFT_FILE

    def prepare(box: FakeBox, role: str) -> None:
        referee_box(box, speedup=1.01, head=json.loads(draft_file.read_text())["sha"])

    factory = FakeBoxFactory(prepare=prepare)
    models = [FakeChatModel([tool_call("submit_proposal", GOOD)])]
    monkeypatch.setattr(sail_box, "SailBoxFactory", lambda cfg: factory)
    monkeypatch.setattr(sail_model, "SailChatModel", lambda worker: models.pop(0))

    args = ["auto", f"file://{src}", "--width", "4", "--rounds", "2", "--template", str(PILOT)]
    # A second target family can be taken in beside an existing one without
    # overwriting its config or documents: both go where these flags say.
    args += ["--configs", "configs/seeded", "--docs-root", "configs/seeded/docs"]
    assert main([*args, "--yes", "--until", "calibrated"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "stopped at ready to run" in out

    cfg = load_config(tmp_path / "configs" / "seeded" / "repo_w4.toml")
    assert (cfg.width, cfg.rounds) == (4, 2)
    assert cfg.target.uncalibrated == () and cfg.target.docs
    assert all(d.startswith("configs/seeded/docs/") for d in cfg.target.docs)
    assert all((tmp_path / d).is_file() for d in cfg.target.docs)
    assert len(factory.created) == 3 and all(b.terminated for b in factory.created)
    assert models == []
