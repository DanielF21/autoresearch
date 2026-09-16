"""``python -m alphaevolve``: run, launch, curve.

``run`` creates boxes and calls the model. ``launch`` starts a run on the
control box ``autoresearch deploy`` made, and the harness's ``fetch``,
``remote-status`` and ``status`` read the result, since the records are the
harness's. ``curve`` reads run directories and spends nothing.

Every command but ``curve`` turns SIGTERM and SIGHUP into a clean exit, as the
harness's commands do, so a killed run still terminates its boxes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alphaevolve import config as config_mod
from autoresearch import shutdown


def cmd_run(args: argparse.Namespace) -> int:
    from alphaevolve import base, budget
    from alphaevolve.worker import EvolveWorker
    from autoresearch import env, observe
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.cli import _refuse_uncalibrated
    from autoresearch.model.sail_model import SailChatModel
    from autoresearch.orchestrator import run as run_mod
    from autoresearch.orchestrator.round import profile_docs_from_repo

    evolve_cfg = config_mod.load(Path(args.config))
    cfg = evolve_cfg.run
    if rc := _refuse_uncalibrated(cfg, "run"):
        return rc
    env.load_dotenv()
    root = Path(args.repo_root).resolve()
    runs_root = Path(args.runs_root) if args.runs_root else cfg.runs_root
    docs = profile_docs_from_repo(root, cfg)
    paths = run_mod.init_run(cfg, runs_root / cfg.run_id, docs)
    boxes = SailBoxFactory(cfg)
    state = base.prepare(paths, evolve_cfg, boxes, runs_root)
    worker = EvolveWorker(
        SailChatModel(cfg.worker), evolve_cfg, paths, state, observe.build_tracer(cfg)
    )
    last = run_mod.run(
        cfg,
        paths,
        worker,
        boxes,
        until_round=args.until,
        log=paths.root / "run.log",
        stop=budget.stop_at(state.budget),
    )
    spent = budget.primary(budget.run_usage(paths))
    print(
        f"completed through round {last} of at most {cfg.rounds} in {paths.root}; "
        f"{spent:,} of a {state.budget:,} token budget spent"
    )
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    from autoresearch.control import commands as control
    from autoresearch.control import deploy as ctl

    # Refused here, before anything starts, if it is not an AlphaEvolve config.
    config_mod.load(Path(args.config))
    cfg, box = control._control_box(args.config)
    cmd = ctl.launch(
        cfg,  # type: ignore[arg-type]
        args.config,
        box,  # type: ignore[arg-type]
        control._launch_env(),
        args.until,
        keep_control=args.keep_control,
        program="alphaevolve",
    )
    print(f"started in the control box: {cmd}")
    print("read progress with: autoresearch remote-status " + args.config)
    return 0


def cmd_curve(args: argparse.Namespace) -> int:
    from alphaevolve import curve

    dirs = [Path(d) for d in args.run_dirs]
    labels = args.label or [d.name for d in dirs]
    if len(labels) != len(dirs):
        print("give one --label per run directory, or none", file=sys.stderr)
        return 2
    curves = {label: curve.points(d) for label, d in zip(labels, dirs, strict=True)}
    empty = [label for label, pts in curves.items() if len(pts) < 2]
    if empty:
        print(f"no completed rounds in: {', '.join(empty)}", file=sys.stderr)
        return 2
    budget = args.budget or min(pts[-1].tokens for pts in curves.values())
    cut = {label: curve.cut(pts, budget) for label, pts in curves.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curve.write_csv(cut, out / "curve.csv")
    curve.plot(cut, out / "curve.png", budget)
    print(f"budget {budget:,} tokens, prompt plus completion")
    for label, pts in cut.items():
        last = pts[-1]
        best = "none" if last.best is None else f"{last.best:.3f}x"
        print(
            f"{label}: best {best} by round {last.round}, {last.tokens:,} tokens, "
            f"{last.uncached:,} uncached, {last.measurements} measurements"
        )
    print(f"wrote {out / 'curve.csv'} and {out / 'curve.png'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphaevolve", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser(
        "run", help="run batches until the token budget (creates boxes, calls the model)"
    )
    p.add_argument("config")
    p.add_argument("--runs-root", default="", help="override the runs root from the config")
    p.add_argument("--repo-root", default=".", help="where config docs paths are resolved")
    p.add_argument("--until", type=int, default=None, help="stop after this batch")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("launch", help="start a run inside the control box")
    p.add_argument("config", help="config path relative to the repo root, as uploaded")
    p.add_argument("--until", type=int, default=None, help="stop after this batch")
    p.add_argument(
        "--keep-control",
        action="store_true",
        help="leave the control box running after the run ends (it never sleeps)",
    )
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("curve", help="tokens against best speedup for run directories (free)")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--label", action="append", default=[], help="one per run directory")
    p.add_argument("--budget", type=int, default=0, help="default: the smallest run's total")
    p.add_argument("--out", default="artifacts/headtohead")
    p.set_defaults(func=cmd_curve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command != "curve":
        shutdown.exit_cleanly_on_signals()
    result: int = args.func(args)
    return result
