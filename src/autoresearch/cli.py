"""Command line entry point.

Local commands, which never touch Sail: ``status``, ``next``, ``intake``.
Commands that create boxes or call the model: ``intake propose``, ``check``,
``profile``, ``calibrate``, ``run``, ``measure``, ``scribe``, and ``auto --yes``.
Control box commands: ``deploy``, ``launch``, ``remote-status``, ``fetch``,
and ``release-control``, which a launch runs after its run ends.
The backstop for a process that died without cleanup: ``reap``.

Every command but ``status`` turns SIGTERM and SIGHUP into a clean exit, so a
killed process still terminates the boxes it holds. See ``shutdown``.

The order for a new target, each step its own command so each cost is its own
decision: ``intake`` clones and derives a draft for free; ``intake propose``
has one model conversation choose the call and inputs and writes the config;
``check`` admits it on one referee box against ``referee/admissibility.py``;
``profile`` writes the worker's documents from one referee box; ``calibrate``
writes the noise floors from one referee box; then ``run``; then ``scribe``
writes the pull request. ``next`` says which step a config is at.

``auto <url>`` runs that whole order. Without ``--yes`` it takes the repository
in for free, prints what every later step creates, and stops. With ``--yes`` it
runs each step in turn, reading the stage from disk before each one, so a second
invocation resumes where the first stopped. A failed check goes back to the
proposing conversation with its report, at most twice. See ``pipeline``.

Every command is a function taking parsed arguments, so the wiring is testable
without a subprocess.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from autoresearch import env, history, pipeline, shutdown
from autoresearch.config import RunConfig, load_config
from autoresearch.orchestrator import status as status_mod


def _load(path: str) -> RunConfig:
    return load_config(Path(path))


def _resolve_run(arg: str) -> tuple[RunConfig, history.RunPaths]:
    """``arg`` is a run directory or a config file; either way, both come back."""
    p = Path(arg)
    if p.is_dir():
        cfg = load_config(p / history.CONFIG_FILE)
        return cfg, history.RunPaths(p)
    cfg = _load(arg)
    return cfg, history.RunPaths(cfg.run_dir)


# ----- commands ----------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    cfg, paths = _resolve_run(args.run)
    if args.runs_root:
        paths = history.RunPaths(Path(args.runs_root) / cfg.run_id)
    st = status_mod.compute_status(paths, cfg.rounds, cfg.worker.model, cfg.run_id)
    print(st.render())
    return 0


def _refuse_uncalibrated(cfg: RunConfig, what: str) -> int:
    """Non zero if any input has no noise floor, with the inputs named."""
    if not cfg.target.uncalibrated:
        return 0
    print(
        f"cannot {what}: no noise floor for {', '.join(cfg.target.uncalibrated)}. "
        "Run autoresearch calibrate on this config first; it writes the floors in.",
        file=sys.stderr,
    )
    return 2


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    if rc := _refuse_uncalibrated(cfg, "run"):
        return rc

    from autoresearch import observe
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.model.sail_model import SailChatModel
    from autoresearch.orchestrator import run as run_mod
    from autoresearch.orchestrator.round import profile_docs_from_repo
    from autoresearch.worker.agent_loop import AgentLoopWorker

    env.load_dotenv()
    root = Path(args.repo_root).resolve()
    run_dir = Path(args.runs_root) / cfg.run_id if args.runs_root else cfg.run_dir
    docs = profile_docs_from_repo(root, cfg)
    paths = run_mod.init_run(cfg, run_dir, docs)
    boxes = SailBoxFactory(cfg)
    worker = AgentLoopWorker(SailChatModel(cfg.worker), boxes, cfg, observe.build_tracer(cfg))
    log = paths.root / "run.log"
    last = run_mod.run(cfg, paths, worker, boxes, until_round=args.until, log=log)
    print(f"completed through round {last} of {cfg.rounds} in {paths.root}")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Phase 2b check 1: measure hand written patches on one real referee box."""
    cfg = _load(args.config)
    if rc := _refuse_uncalibrated(cfg, "measure"):
        return rc

    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.referee.referee import Referee

    env.load_dotenv()
    boxes = SailBoxFactory(cfg)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    box = boxes.create(name=f"measure-{ts}", role="referee")
    print(f"referee box {box.name} ({box.box_id})", flush=True)
    out_dir = Path(args.out) / "measure"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    try:
        ref = Referee(box, cfg)
        ref.setup()
        for patch_path in args.patches:
            patch = Path(patch_path).read_text()
            print(f"measuring {patch_path} ...", flush=True)
            m = ref.measure(patch)
            report.append({"patch": patch_path, "measurement": m.to_dict()})
            tests = (
                ", ".join(f"{t.scope} {'pass' if t.ok else 'FAIL'}" for t in m.tests) or "not run"
            )
            per_input = ", ".join(
                f"{i.name} {'--' if i.speedup is None else format(i.speedup, '.3f')}"
                for i in m.inputs
            )
            print(
                f"  applied {m.applied}; tests {tests}; result matches {m.result_matches}; "
                f"geomean {m.speedup}; worst {m.worst_speedup}; "
                f"slower on {list(m.regressions) or 'nothing'}; clears noise {m.clears_noise}; "
                f"errors {list(m.errors)}; {m.wall_s:.0f}s\n"
                f"  per input: {per_input or 'none timed'}",
                flush=True,
            )
            if ref.broken:
                print(f"  referee marked broken: {ref.broken}", flush=True)
                break
    finally:
        (out_dir / f"{ts}.json").write_text(json.dumps(report, indent=2) + "\n")
        if not args.keep:
            box.terminate()
            print("box terminated", flush=True)
    print(f"wrote {out_dir / f'{ts}.json'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Survey a target on one referee box and judge it against the admissibility rules."""
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.referee import admissibility
    from autoresearch.referee.referee import Referee

    env.load_dotenv()
    cfg = _load(args.config)
    boxes = SailBoxFactory(cfg)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    box = boxes.create(name=f"check-{ts}", role="referee")
    print(f"referee box {box.name} ({box.box_id})", flush=True)
    try:
        ref = Referee(box, cfg)
        ref.setup()
        survey = ref.survey()
        if ref.broken:
            print(f"referee marked broken: {ref.broken}", file=sys.stderr)
    finally:
        if not args.keep:
            box.terminate()
            print("box terminated", flush=True)
    verdicts = admissibility.judge(
        cfg.target, survey, cfg.referee.repeats_per_launch, cfg.referee.pairs
    )
    report = admissibility.render(cfg.target, survey, verdicts)
    print(report)
    out_dir = Path(args.out) / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "config": args.config,
        "sha": cfg.target.sha,
        "target_hash": admissibility.admission_hash(cfg.target),
        "box_id": box.box_id,
        "at": ts,
        "inputs": [i.__dict__ for i in survey.inputs],
        "tests": [t.to_dict() for t in survey.tests],
        "errors": list(survey.errors),
        "verdicts": [v.__dict__ for v in verdicts],
        "report": report,
    }
    (out_dir / f"{ts}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nrecord: {out_dir / f'{ts}.json'}")
    return 1 if any(v.failed for v in verdicts) else 0


def cmd_intake(args: argparse.Namespace) -> int:
    """``intake <url>`` clones and drafts for free; ``intake propose <dir>`` calls the model."""
    if args.what == "propose":
        if not args.draft:
            print("usage: autoresearch intake propose runs/intake/<name>", file=sys.stderr)
            return 2
        return _intake_propose(args)
    if args.draft:
        print(f"unexpected argument {args.draft!r}; did you mean intake propose?", file=sys.stderr)
        return 2
    return _intake_clone(args)


def _intake_clone(args: argparse.Namespace) -> int:
    from autoresearch.intake import derive, scope

    url = args.what
    template = _shaped(_load(args.template), args)
    try:
        out = Path(args.root) / scope.repo_name(url)
        repo = out / derive.REPO_DIR
        sha = scope.clone(url, repo)
    except scope.IntakeError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 2
    try:
        draft, findings = derive.derive(url, repo, sha, args.template, template.width, args.package)
    except scope.IntakeError as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 2
    for f in findings:
        print(f"{f.level.upper():<6} {f.rule}: {f.detail}")
    if draft is None:
        print(
            f"\nnot in scope: {url}. Nothing was drafted. The clone is left at {repo} to read; "
            f"remove {out} before taking it in again."
        )
        return 2
    derive.write_draft(out, repo, draft)
    print(
        f"\n{draft.name} at {draft.sha[:12]}: package {draft.package} from {draft.package_root}, "
        f"whole suite {draft.tests_full}\n"
        f"pip {', '.join(draft.pip) or 'none'}; may change {', '.join(draft.allow)}; "
        f"never {', '.join(draft.deny) or 'n/a'}\n"
        f"{len(draft.benchmarks)} benchmark sources found\n\n"
        f"draft:   {out / derive.DRAFT_FILE}\n"
        f"brief:   {out / derive.BRIEF_FILE}\n\n"
        f"next:    autoresearch intake propose {out}\n"
        f"creates: no box; one model conversation with {template.worker.model}, "
        f"at most {args.max_turns} replies"
    )
    return 0


def _shaped(template: RunConfig, args: argparse.Namespace) -> RunConfig:
    """The template with ``--width`` and ``--rounds`` applied where given."""
    from dataclasses import replace

    width = getattr(args, "width", 0) or template.width
    rounds = getattr(args, "rounds", 0) or template.rounds
    return replace(template, width=width, rounds=rounds)


def _intake_propose(args: argparse.Namespace, report: str = "") -> int:
    """A proposal from a new conversation, or with ``report``, the saved one continued."""
    from dataclasses import replace

    from autoresearch.intake import derive, propose
    from autoresearch.model.protocol import ModelError
    from autoresearch.model.sail_model import SailChatModel

    out = Path(args.draft)
    try:
        draft = derive.load_draft(out)
    except (OSError, ValueError, KeyError) as e:
        print(f"no draft in {out}: {e}", file=sys.stderr)
        return 2
    template = _shaped(_load(draft.template), args)
    config_path = Path(args.configs) / f"{draft.run_id}.toml"
    if config_path.exists():
        print(f"{config_path} exists; move it aside first. No model was called.", file=sys.stderr)
        return 2
    env.load_dotenv()
    worker = replace(template.worker, model=args.model or template.worker.model)
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verb = "repairing the proposal" if report else "proposing"
    print(
        f"{verb} for {draft.name} with {worker.model}, at most {args.max_turns} replies",
        flush=True,
    )
    try:
        if report:
            outcome = propose.repair(
                SailChatModel(worker),
                out,
                draft,
                template,
                report,
                when=when,
                max_turns=args.max_turns,
            )
        else:
            outcome = propose.propose(
                SailChatModel(worker),
                out,
                draft,
                template,
                (out / derive.BRIEF_FILE).read_text(),
                when=when,
                max_turns=args.max_turns,
            )
    except (ModelError, propose.ProposeError) as e:
        print(f"stopped: {e}\nmessages and usage: {out}", file=sys.stderr)
        return 1
    if config_path.exists():
        print(f"{config_path} appeared meanwhile; not overwritten", file=sys.stderr)
        return 1
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(outcome.text)
    (out / "proposal.json").write_text(json.dumps(outcome.proposal.to_dict(), indent=2) + "\n")
    u = outcome.usage
    print(
        f"accepted after {outcome.turns} replies; tokens: {u.prompt_tokens} prompt "
        f"({u.cached_tokens} cached), {u.completion_tokens} completion\n"
        f"axis: {outcome.proposal.axis}\n"
        f"inputs: {', '.join(i.name for i in outcome.proposal.inputs)}\n\n"
        f"config:  {config_path}\n\n"
        f"next:    autoresearch check {config_path}\n"
        f"creates: 1 referee box, size {template.boxes.referee_size}, no model call"
    )
    return 0


def _write_back(path: Path, original: str, edited: str) -> bool:
    """Write ``edited`` over ``path`` unless the file changed since ``original`` was read.

    A box step can take an hour. A hand edit made meanwhile must not be lost to
    a write computed from the text as it was before.
    """
    if path.read_text() != original:
        print(
            f"not written: {path} changed while the box ran. Nothing was lost; "
            "the record above has every number.",
            file=sys.stderr,
        )
        return False
    path.write_text(edited)
    return True


def cmd_profile(args: argparse.Namespace) -> int:
    """Profile every input on one referee box and write the worker's documents."""
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.config_edit import set_docs
    from autoresearch.referee import profile_docs
    from autoresearch.referee.referee import Referee

    env.load_dotenv()
    config_path = Path(args.config)
    original = config_path.read_text()
    cfg = _load(args.config)
    boxes = SailBoxFactory(cfg)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    box = boxes.create(name=f"profile-{ts}", role="referee")
    print(f"referee box {box.name} ({box.box_id})", flush=True)
    try:
        ref = Referee(box, cfg)
        ref.setup()
        rows = ref.profile()
        if ref.broken:
            print(f"referee marked broken: {ref.broken}", file=sys.stderr)
    finally:
        if not args.keep:
            box.terminate()
            print("box terminated", flush=True)
    for r in rows:
        print(f"{r.name:<16} call {r.call_s:.4f}s  hot share {100 * r.share:.1f}%")
    written = profile_docs.write_documents(
        Path(args.docs_root) / cfg.run_id, profile_docs.documents(cfg.target, rows)
    )
    for p in written:
        print(f"wrote {p}")
    if args.no_write:
        return 0
    if not _write_back(config_path, original, set_docs(original, [str(p) for p in written])):
        return 1
    print(f"docs set in {config_path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """A noise floor per input from one referee box, written into the config."""
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.config_edit import set_noise_floors
    from autoresearch.referee import calibration
    from autoresearch.referee.referee import Referee

    env.load_dotenv()
    config_path = Path(args.config)
    original = config_path.read_text()
    cfg = _load(args.config)
    only = tuple(i.name for i in cfg.target.inputs if i.name not in args.skip)
    if not only:
        print("every input was skipped", file=sys.stderr)
        return 2
    per_input = args.rounds * cfg.referee.pairs
    false_alarm = calibration.FALSE_ALARM
    print(
        f"calibrating {len(only)} inputs at {per_input} pairs each "
        f"({args.rounds} rounds of {cfg.referee.pairs}), floor at 1 in "
        f"{int(1 / false_alarm)}: {', '.join(only)}",
        flush=True,
    )

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = SailBoxFactory(cfg)
    box = boxes.create(name=f"calibrate-{ts}", role="referee")
    print(f"referee box {box.name} ({box.box_id})", flush=True)

    def progress(name: str, done: int) -> None:
        print(f"  {name}: {done}/{per_input} pairs", flush=True)

    try:
        ref = Referee(box, cfg)
        ref.setup()
        measured = ref.null_pairs(rounds=args.rounds, only=only, progress=progress)
        if ref.broken:
            print(f"referee marked broken: {ref.broken}", file=sys.stderr)
    finally:
        if not args.keep:
            box.terminate()
            print("box terminated", flush=True)

    lines: list[str] = []
    records: list[dict[str, object]] = []
    for name in only:
        line, rec = calibration.report(name, measured[name], cfg.referee.pairs, false_alarm)
        lines.append(line)
        records.append(rec)
    print("\n" + "\n".join(lines))

    record = out_dir / f"{ts}.jsonl"
    header = {
        "config": args.config,
        "sha": cfg.target.sha,
        "box_id": box.box_id,
        "rounds": args.rounds,
        "pairs": cfg.referee.pairs,
        "repeats_per_launch": cfg.referee.repeats_per_launch,
        "false_alarm": false_alarm,
        "at": ts,
    }
    calibration.write_record(record, header, records, measured)
    print(f"\nrecord: {record}")

    floors = {
        str(rec["input"]): (float(str(rec["floor"])), f"calibrated {ts}, {rec['clean']} null pairs")
        for rec in records
        if rec["floor"] is not None
    }
    if not floors:
        print("no input produced a floor; nothing to write", file=sys.stderr)
        return 1
    if args.no_write:
        for name, (floor, comment) in floors.items():
            print(f"{name}: noise_floor = {floor:.4f}   # {comment}")
        return 0
    if not _write_back(config_path, original, set_noise_floors(original, floors)):
        return 1
    print(f"floors for {', '.join(floors)} written into {config_path}")
    return 0 if len(floors) == len(only) else 1


def cmd_next(args: argparse.Namespace) -> int:
    """Which step a target is at, and the one command that comes next. Free."""
    from autoresearch.pipeline import Stage, config_stage

    cfg = _load(args.config)
    found = config_stage(cfg, Path(args.out), Path(args.repo_root))
    referee = f"1 referee box, size {cfg.boxes.referee_size}, no model call"

    def say(stage: str, command: str, creates: str) -> int:
        print(f"stage:   {stage}\nnext:    {command}\ncreates: {creates}")
        return 0

    if found.stage is Stage.NOT_CHECKED:
        return say("not checked", f"autoresearch check {args.config}", referee)
    if found.stage is Stage.CHECK_FAILED:
        print(
            f"stage:   check failed in {found.record}: {', '.join(found.failed)}\n"
            "next:    change the config and check again, or drop the target"
        )
        return 1
    if found.stage is Stage.NEEDS_PROFILE:
        return say(f"checked in {found.record}", f"autoresearch profile {args.config}", referee)
    if found.stage is Stage.NEEDS_CALIBRATE:
        return say(
            f"profiled; no floor for {', '.join(found.uncalibrated)}",
            f"autoresearch calibrate {args.config} --rounds 7",
            referee,
        )
    return say(
        "ready to run",
        f"autoresearch run {args.config} --until 1",
        f"{cfg.width} worker boxes (size {cfg.boxes.worker_size}) and {cfg.width} referee "
        f"boxes (size {cfg.boxes.referee_size}); {cfg.width} model attempts per round "
        f"with {cfg.worker.model}",
    )


def cmd_scribe(args: argparse.Namespace) -> int:
    """Pick a finished run's candidate and write its pull request. One model conversation."""
    from autoresearch.scribe import cli as scribe_cli

    argv = [args.run_dir, "--n", str(args.n), "--out", args.out]
    for flag, value in (
        ("--repo", args.repo),
        ("--prs", args.prs),
        ("--model", args.model),
        ("--source", args.source),
    ):
        if value:
            argv += [flag, value]
    return scribe_cli.main(argv)


def _auto_steps() -> pipeline.Steps:
    """Each stage of ``auto`` as the command that does it alone, with its arguments built."""
    from autoresearch.intake.derive import REPO_DIR
    from autoresearch.orchestrator.run import RunError
    from autoresearch.scribe import cli as scribe_cli

    def intake(ctx: pipeline.Context) -> int:
        return _intake_clone(
            argparse.Namespace(
                what=ctx.url,
                template=str(ctx.template),
                package=ctx.package,
                root=str(ctx.intake_root),
                width=ctx.width,
                rounds=ctx.rounds,
                max_turns=pipeline.PROPOSE_TURNS,
            )
        )

    def proposal(ctx: pipeline.Context, report: str = "") -> int:
        ns = argparse.Namespace(
            draft=str(ctx.draft_dir),
            configs=str(ctx.configs),
            model="",
            max_turns=pipeline.PROPOSE_TURNS,
            width=ctx.width,
            rounds=ctx.rounds,
        )
        return _intake_propose(ns, report)

    def config(ctx: pipeline.Context) -> str:
        return str(ctx.config_path())

    def run(ctx: pipeline.Context) -> int:
        ns = argparse.Namespace(
            config=config(ctx),
            runs_root=str(ctx.runs_root),
            repo_root=str(ctx.repo_root),
            until=None,
        )
        try:
            return cmd_run(ns)
        except RunError as e:
            print(f"run stopped: {e}", file=sys.stderr)
            return 1

    return pipeline.Steps(
        intake=intake,
        propose=proposal,
        repair=proposal,
        check=lambda ctx: cmd_check(
            argparse.Namespace(config=config(ctx), out=str(ctx.runs_root), keep=False)
        ),
        profile=lambda ctx: cmd_profile(
            argparse.Namespace(
                config=config(ctx), docs_root=str(ctx.docs_root), no_write=False, keep=False
            )
        ),
        calibrate=lambda ctx: cmd_calibrate(
            argparse.Namespace(
                config=config(ctx),
                rounds=pipeline.CALIBRATE_ROUNDS,
                skip=[],
                out=str(ctx.calibration_out),
                no_write=False,
                keep=False,
            )
        ),
        run=run,
        scribe=lambda ctx: scribe_cli.main(
            [
                str(ctx.run_dir()),
                "--source",
                str(ctx.draft_dir / REPO_DIR),
                "--out",
                str(ctx.scribe_root),
            ]
        ),
    )


def cmd_auto(args: argparse.Namespace) -> int:
    """A repository URL to a patch and a pull request, every step in order, resumable."""
    ctx = pipeline.Context(
        url=args.url,
        width=args.width,
        rounds=args.rounds,
        template=Path(args.template),
        package=args.package,
        intake_root=Path(args.root),
        configs=Path(args.configs),
        docs_root=Path(args.docs_root),
    )
    if args.yes:
        env.load_dotenv()
    return pipeline.auto(ctx, _auto_steps(), yes=args.yes, until=args.until)


def cmd_reap(args: argparse.Namespace) -> int:
    """List every live box in the app; terminate them with --yes.

    The backstop for what no process can clean up after itself: ``kill -9``, a
    dead laptop, a lost control box, a create that raised and came up later.
    """
    from autoresearch.boxes import sail_box
    from autoresearch.boxes.protocol import BoxError

    env.load_dotenv()
    live = sail_box.live_boxes(prefix=args.prefix)
    if not live:
        print("no live boxes")
        return 0
    for b in live:
        print(f"{b.name}  {b.box_id}  {b.status}  created {b.created_at}")
    if not args.yes:
        print(
            f"\n{len(live)} live, nothing terminated. --yes terminates every one listed, "
            "including any run still going."
        )
        return 0
    failed = 0
    for b in live:
        try:
            sail_box.terminate_box(b.box_id)
        except BoxError as e:
            failed += 1
            print(f"did not terminate {b.name} ({b.box_id}): {e}", file=sys.stderr)
            continue
        print(f"terminated {b.name}")
    return 1 if failed else 0


# ----- parser ------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("status", help="totals for a run directory or config")
    p.add_argument("run", help="run directory, or config file")
    p.add_argument("--runs-root", default="", help="override the runs root from the config")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run", help="run rounds of one setting (creates boxes, calls the model)")
    p.add_argument("config")
    p.add_argument("--runs-root", default="", help="override the runs root from the config")
    p.add_argument("--repo-root", default=".", help="where config docs paths are resolved")
    p.add_argument("--until", type=int, default=None, help="stop after this round")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("measure", help="measure patch files on one real referee box")
    p.add_argument("config")
    p.add_argument("patches", nargs="+")
    p.add_argument("--out", default="runs", help="where the report is written")
    p.add_argument("--keep", action="store_true", help="leave the box running")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser(
        "check", help="survey a target on one referee box and judge whether it can be measured"
    )
    p.add_argument("config")
    p.add_argument("--out", default="runs", help="where the report is written")
    p.add_argument("--keep", action="store_true", help="leave the box running")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser(
        "intake",
        help="intake <url>: clone, judge scope, draft (free); "
        "intake propose <dir>: one model conversation writes the config",
    )
    p.add_argument("what", help="a repository URL, or 'propose'")
    p.add_argument("draft", nargs="?", default="", help="with propose: runs/intake/<name>")
    p.add_argument(
        "--template", default="configs/t1_w4d.toml", help="source of every non target section"
    )
    p.add_argument("--package", default="", help="the package, when the repo has several")
    p.add_argument("--root", default="runs/intake", help="where clones and drafts go")
    p.add_argument("--model", default="", help="propose only: default the template's worker model")
    p.add_argument(
        "--max-turns", type=int, default=40, help="propose only: replies before giving up"
    )
    p.add_argument("--configs", default="configs", help="propose only: where the config is written")
    p.add_argument("--width", type=int, default=0, help="run width; default the template's")
    p.add_argument("--rounds", type=int, default=0, help="propose only: default the template's")
    p.set_defaults(func=cmd_intake)

    p = sub.add_parser(
        "auto",
        help="a repository URL to a patch and a pull request: every step in order; "
        "prints the plan and spends nothing without --yes",
    )
    p.add_argument("url")
    p.add_argument("--width", type=int, required=True, help="worker attempts per round")
    p.add_argument("--rounds", type=int, required=True, help="rounds in the run")
    p.add_argument("--yes", action="store_true", help="create the boxes and call the model")
    p.add_argument(
        "--until",
        default="",
        choices=["", *pipeline.UNTIL],
        help="stop once this stage is reached: config, admitted, calibrated, run",
    )
    p.add_argument("--package", default="", help="the package, when the repo has several")
    p.add_argument(
        "--template", default="configs/t1_w4d.toml", help="source of every non target section"
    )
    p.add_argument("--root", default="runs/auto", help="where clones and drafts go")
    p.add_argument("--configs", default="configs", help="where the config is written")
    p.add_argument("--docs-root", default="configs/docs", help="documents go in <this>/<run_id>")
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser(
        "scribe", help="pick a finished run's candidate and write its pull request (model call)"
    )
    p.add_argument("run_dir")
    p.add_argument("--repo", default="", help="owner/name; default: the run's target repo")
    p.add_argument("--n", type=int, default=20, help="merged pull requests to show")
    p.add_argument("--prs", default="", help="a saved prs.json to reuse instead of fetching")
    p.add_argument("--model", default="", help="default: the run's worker model")
    p.add_argument("--out", default="runs/scribe")
    p.add_argument("--source", default="", help="a checkout at the run's sha")
    p.set_defaults(func=cmd_scribe)

    p = sub.add_parser(
        "profile", help="profile every input on one referee box and write the worker's documents"
    )
    p.add_argument("config")
    p.add_argument("--docs-root", default="configs/docs", help="documents go in <this>/<run_id>")
    p.add_argument("--no-write", action="store_true", help="leave the config's docs unchanged")
    p.add_argument("--keep", action="store_true", help="leave the box running")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser(
        "calibrate", help="measure a noise floor per input on one referee box, into the config"
    )
    p.add_argument("config", help="the config whose inputs are calibrated")
    p.add_argument("--rounds", type=int, default=7, help="passes of [referee].pairs per input")
    p.add_argument("--skip", action="append", default=[], help="an input to leave out; repeatable")
    p.add_argument("--out", default="artifacts/calibration", help="where the record goes")
    p.add_argument("--no-write", action="store_true", help="print the floors, leave the config")
    p.add_argument("--keep", action="store_true", help="leave the box running")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("next", help="which step a target is at and what the next one creates")
    p.add_argument("config")
    p.add_argument("--out", default="runs", help="where check records were written")
    p.add_argument("--repo-root", default=".", help="where config docs paths are resolved")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("reap", help="list every live box in the app, and terminate them with --yes")
    p.add_argument("--prefix", default="", help="only boxes whose name starts with this")
    p.add_argument("--yes", action="store_true", help="terminate every box listed")
    p.set_defaults(func=cmd_reap)

    try:
        from autoresearch.control import commands as control

        control.register(sub)
    except ImportError:
        pass
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command != "status":
        shutdown.exit_cleanly_on_signals()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
