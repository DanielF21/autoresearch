"""The stage a target is at, read from disk, and ``auto``, which runs every stage in order.

Every stage leaves a file behind: the draft, the config, a check record matching
the config's admission hash, the docs and floors written into the config, the
run's rounds, the Scribe's pick. ``where`` reads those files and nothing else, so
``next`` and ``auto`` agree on the stage, and ``auto`` resumes from whatever an
earlier invocation left, however it ended.

``auto`` runs the step for the stage it finds, then reads the stage again. A step
that leaves the stage where it was stops the command, so a failing step never
loops. A check that fails goes back to the proposing conversation with its
report, at most ``MAX_REPAIRS`` times; the rejected config is kept.
"""

from __future__ import annotations

import enum
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from autoresearch import history
from autoresearch.config import RunConfig, load_config
from autoresearch.intake import derive, scope
from autoresearch.referee import admissibility

MAX_REPAIRS = 2
PROPOSE_TURNS = 40
CALIBRATE_ROUNDS = 7
# The width experiment on networkx, 2026-09-13: about $1.2k of inference over about
# 775 attempts. Printed as that target's rate, never as this one's.
NETWORKX_DOLLARS_PER_ATTEMPT = 1.55
REPAIRS_FILE = "repairs.json"
REJECTED_DIR = "rejected"
PICK_FILE = "pick.json"


class Stage(enum.StrEnum):
    NEEDS_INTAKE = "needs intake"
    NEEDS_CONFIG = "needs config"
    NOT_CHECKED = "not checked"
    CHECK_FAILED = "check failed"
    NEEDS_PROFILE = "needs profile"
    NEEDS_CALIBRATE = "needs calibrate"
    READY = "ready to run"
    RUN_INCOMPLETE = "run incomplete"
    NEEDS_SCRIBE = "needs scribe"
    DONE = "done"


ORDER: tuple[Stage, ...] = tuple(Stage)

# ``--until X`` stops once the stage has reached the one named here.
UNTIL: dict[str, Stage] = {
    "config": Stage.NOT_CHECKED,
    "admitted": Stage.NEEDS_PROFILE,
    "calibrated": Stage.READY,
    "run": Stage.NEEDS_SCRIBE,
}


@dataclass(frozen=True)
class Where:
    stage: Stage
    record: Path | None = None
    failed: tuple[str, ...] = ()
    report: str = ""
    uncalibrated: tuple[str, ...] = ()
    rounds_done: int = 0


def _report_from(rec: dict[str, object]) -> str:
    """The check's printed report, or its verdict lines for a record written before it was kept."""
    report = rec.get("report")
    if isinstance(report, str) and report:
        return report
    verdicts = rec.get("verdicts")
    return "\n".join(
        f"{str(v.get('level', '')).upper():<5} {v.get('rule')}  {v.get('detail', '')}"
        for v in (verdicts if isinstance(verdicts, list) else [])
        if isinstance(v, dict)
    )


def config_stage(cfg: RunConfig, runs_root: Path, repo_root: Path) -> Where:
    """Checked, profiled, calibrated: which of these a config has, from its check records."""
    target = cfg.target
    wanted = admissibility.admission_hash(target)
    matching: list[tuple[Path, dict[str, object]]] = []
    for p in sorted((runs_root / "check").glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if rec.get("sha") == target.sha and rec.get("target_hash") == wanted:
            matching.append((p, rec))
    if not matching:
        return Where(Stage.NOT_CHECKED)
    path, rec = matching[-1]
    verdicts = rec.get("verdicts")
    failed = tuple(
        str(v.get("rule"))
        for v in (verdicts if isinstance(verdicts, list) else [])
        if isinstance(v, dict) and v.get("level") == "fail"
    )
    if failed:
        return Where(Stage.CHECK_FAILED, path, failed, _report_from(rec))
    if not target.docs or any(not (repo_root / d).is_file() for d in target.docs):
        return Where(Stage.NEEDS_PROFILE, path)
    if target.uncalibrated:
        return Where(Stage.NEEDS_CALIBRATE, path, uncalibrated=target.uncalibrated)
    return Where(Stage.READY, path)


def latest_scribe(scribe_root: Path, run_id: str) -> Path | None:
    """The newest Scribe output for a run that got as far as a pick, or None."""
    done = sorted(p.parent for p in (scribe_root / run_id).glob(f"*/{PICK_FILE}"))
    return done[-1] if done else None


def run_stage(cfg: RunConfig, runs_root: Path, scribe_root: Path) -> Where:
    paths = history.RunPaths(runs_root / cfg.run_id)
    if not paths.config.exists():
        return Where(Stage.READY)
    done = history.last_completed_round(paths)
    if done < cfg.rounds:
        return Where(Stage.RUN_INCOMPLETE, rounds_done=done)
    if latest_scribe(scribe_root, cfg.run_id) is None:
        return Where(Stage.NEEDS_SCRIBE, rounds_done=done)
    return Where(Stage.DONE, rounds_done=done)


# ----- auto ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """One ``auto`` invocation: what to take in, at what shape, and where everything goes."""

    url: str
    width: int
    rounds: int
    template: Path
    package: str = ""
    intake_root: Path = Path("runs/auto")
    configs: Path = Path("configs")
    runs_root: Path = Path("runs")
    scribe_root: Path = Path("runs/scribe")
    docs_root: Path = Path("configs/docs")
    calibration_out: Path = Path("artifacts/calibration")
    repo_root: Path = Path()

    @property
    def draft_dir(self) -> Path:
        return self.intake_root / scope.repo_name(self.url)

    def draft(self) -> derive.Draft:
        return derive.load_draft(self.draft_dir)

    def config_path(self) -> Path:
        return self.configs / f"{self.draft().run_id}.toml"

    def run_dir(self) -> Path:
        return self.runs_root / self.draft().run_id


Step = Callable[[Context], int]


@dataclass(frozen=True)
class Steps:
    """What runs each stage. The command line wires in the real commands; tests, fakes."""

    intake: Step
    propose: Step
    repair: Callable[[Context, str], int]
    check: Step
    profile: Step
    calibrate: Step
    run: Step
    scribe: Step


def where(ctx: Context) -> Where:
    if not (ctx.draft_dir / derive.DRAFT_FILE).is_file():
        return Where(Stage.NEEDS_INTAKE)
    config_path = ctx.config_path()
    if not config_path.is_file():
        return Where(Stage.NEEDS_CONFIG)
    cfg = load_config(config_path)
    found = config_stage(cfg, ctx.runs_root, ctx.repo_root)
    if found.stage is not Stage.READY:
        return found
    return run_stage(cfg, ctx.runs_root, ctx.scribe_root)


def shaped(template: RunConfig, width: int, rounds: int) -> RunConfig:
    return replace(template, width=width, rounds=rounds)


def plan(ctx: Context, template: RunConfig) -> str:
    """Everything ``--yes`` may create, stage by stage."""
    t = shaped(template, ctx.width, ctx.rounds)
    b, w = t.boxes, t.worker
    attempts = t.width * t.rounds
    dollars = attempts * NETWORKX_DOLLARS_PER_ATTEMPT
    return "\n".join(
        [
            f"plan for {ctx.url} at width {t.width}, {t.rounds} rounds, model {w.model}:",
            f"  propose    1 model conversation, at most {PROPOSE_TURNS} replies, no box",
            f"  check      1 referee box (size {b.referee_size}); each failed check goes back "
            f"to the conversation, at most {MAX_REPAIRS} times, so at most {MAX_REPAIRS + 1} boxes",
            f"  profile    1 referee box (size {b.referee_size})",
            f"  calibrate  1 referee box (size {b.referee_size}), {CALIBRATE_ROUNDS} rounds",
            f"  run        {t.width} worker boxes (size {b.worker_size}) and {t.width} referee "
            f"boxes per round, {attempts} model attempts of at most {w.max_turns} turns and "
            f"{w.max_seconds}s each",
            "  scribe     1 model conversation, no box",
            f"  cost       about ${dollars:,.0f} of inference for the run at the networkx width "
            f"experiment's ${NETWORKX_DOLLARS_PER_ATTEMPT:.2f} per attempt; never measured on "
            "this target",
        ]
    )


def _repairs(ctx: Context) -> list[dict[str, object]]:
    path = ctx.draft_dir / REPAIRS_FILE
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else []


def _repair(ctx: Context, steps: Steps, found: Where) -> int:
    done = _repairs(ctx)
    if len(done) >= MAX_REPAIRS:
        print(
            f"stopped: check failed again after {len(done)} repairs: {', '.join(found.failed)}. "
            f"Record {found.record}; rejected configs in {ctx.draft_dir / REJECTED_DIR}."
        )
        return 1
    n = len(done) + 1
    config_path = ctx.config_path()
    rejected = ctx.draft_dir / REJECTED_DIR / f"{n}.toml"
    rejected.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(config_path, rejected)
    print(f"check failed: {', '.join(found.failed)}. Repair {n} of {MAX_REPAIRS}.", flush=True)
    rc = steps.repair(ctx, found.report)
    if rc != 0 or not config_path.is_file():
        # Put the checked config back, so the stage stays a failed check and nothing
        # proposes from scratch.
        if config_path.is_file():
            config_path.unlink()
        shutil.move(rejected, config_path)
        return rc or 1
    done.append(
        {"record": str(found.record), "failed": list(found.failed), "rejected": str(rejected)}
    )
    (ctx.draft_dir / REPAIRS_FILE).write_text(json.dumps(done, indent=2) + "\n")
    return 0


def _run_step(ctx: Context, steps: Steps, found: Where) -> int:
    if found.stage is Stage.NEEDS_INTAKE:
        return steps.intake(ctx)
    if found.stage is Stage.NEEDS_CONFIG:
        return steps.propose(ctx)
    if found.stage is Stage.NOT_CHECKED:
        return steps.check(ctx)
    if found.stage is Stage.CHECK_FAILED:
        return _repair(ctx, steps, found)
    if found.stage is Stage.NEEDS_PROFILE:
        return steps.profile(ctx)
    if found.stage is Stage.NEEDS_CALIBRATE:
        return steps.calibrate(ctx)
    if found.stage in (Stage.READY, Stage.RUN_INCOMPLETE):
        return steps.run(ctx)
    return steps.scribe(ctx)


def _advanced(before: Where, after: Where) -> bool:
    if after.stage is not before.stage:
        return True
    return before.stage is Stage.RUN_INCOMPLETE and after.rounds_done > before.rounds_done


def _stall_detail(ctx: Context, found: Where) -> str:
    if found.stage is Stage.NEEDS_CALIBRATE:
        return f"; still no floor for {', '.join(found.uncalibrated)}"
    if found.stage is Stage.NEEDS_SCRIBE:
        return f"; the Scribe wrote no pick for {ctx.run_dir()}"
    if found.stage in (Stage.READY, Stage.RUN_INCOMPLETE):
        return f"; see {ctx.run_dir() / 'run.log'}"
    return ""


def _usage_line(label: str, path: Path) -> str:
    try:
        u = json.loads(path.read_text())
    except (OSError, ValueError):
        return f"  {label}: no usage recorded"
    return (
        f"  {label}: {u.get('prompt_tokens', 0)} prompt tokens ({u.get('cached_tokens', 0)} "
        f"cached), {u.get('completion_tokens', 0)} completion"
    )


def summary(ctx: Context) -> str:
    run_id = ctx.draft().run_id
    out = latest_scribe(ctx.scribe_root, run_id)
    if out is None:
        return f"done: {run_id} has no Scribe output"
    pick = json.loads((out / PICK_FILE).read_text())
    lines = [f"done: {run_id}"]
    if (out / "pr.md").is_file():
        lines += [
            f"  picked attempt {pick.get('attempt')}",
            f"  patch  {out / 'patch.diff'}",
            f"  pr     {out / 'pr.md'}",
        ]
    else:
        lines.append(f"  no attempt picked: {pick.get('reasons')}")
    lines += [
        "tokens:",
        _usage_line("propose", ctx.draft_dir / "usage.json"),
        _usage_line("scribe", out / "usage.json"),
        f"  run: autoresearch status {ctx.run_dir()}",
    ]
    return "\n".join(lines)


def auto(ctx: Context, steps: Steps, *, yes: bool, until: str = "") -> int:
    """Take the repository in for free, print the plan, and with ``yes`` run every stage."""
    if until and until not in UNTIL:
        print(f"--until must be one of {', '.join(UNTIL)}")
        return 2
    try:
        found = where(ctx)
    except scope.IntakeError as e:
        print(f"stopped: {e}")
        return 2
    if found.stage is Stage.NEEDS_INTAKE:
        rc = steps.intake(ctx)
        if rc != 0:
            return rc
    print(plan(ctx, load_config(ctx.template)))
    if not yes:
        print("\nnothing spent. Add --yes to run the stages above.")
        return 0
    stop_at = UNTIL.get(until)
    while True:
        found = where(ctx)
        print(f"\n== stage: {found.stage}", flush=True)
        if found.stage is Stage.DONE:
            print(summary(ctx))
            return 0
        if stop_at is not None and ORDER.index(found.stage) >= ORDER.index(stop_at):
            print(f"stopped at {found.stage}, as --until {until} asked")
            return 0
        rc = _run_step(ctx, steps, found)
        after = where(ctx)
        if not _advanced(found, after):
            print(
                f"stopped: {found.stage} did not advance (step exited {rc})"
                f"{_stall_detail(ctx, after)}"
            )
            return rc or 1
