"""Command line entry point.

Local commands, which never touch Sail: ``status``.
Commands that create boxes or call the model: ``run``, ``measure``, ``check``.
Control box commands: ``deploy``, ``launch``, ``remote-status``, ``fetch``,
and ``release-control``, which a launch runs after its run ends.
The backstop for a process that died without cleanup: ``reap``.

Every command but ``status`` turns SIGTERM and SIGHUP into a clean exit, so a
killed process still terminates the boxes it holds. See ``shutdown``.

``check`` is how a target is admitted: one referee box, no patch, no model.
It surveys the base tree and judges it against the rules in
``referee/admissibility.py``. A run refuses a config with uncalibrated
inputs, so the order for a new target is check, calibrate, run.

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

from autoresearch import env, history, shutdown
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
        "Run calibrate.py on this config first and paste the floors in.",
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
    verdicts = admissibility.judge(cfg.target, survey, cfg.referee.repeats_per_launch)
    print(admissibility.render(cfg.target, survey, verdicts))
    out_dir = Path(args.out) / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "config": args.config,
        "sha": cfg.target.sha,
        "box_id": box.box_id,
        "at": ts,
        "inputs": [i.__dict__ for i in survey.inputs],
        "tests": [t.to_dict() for t in survey.tests],
        "errors": list(survey.errors),
        "verdicts": [v.__dict__ for v in verdicts],
    }
    (out_dir / f"{ts}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nrecord: {out_dir / f'{ts}.json'}")
    return 1 if any(v.failed for v in verdicts) else 0


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
