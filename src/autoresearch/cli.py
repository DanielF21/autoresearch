"""Command line entry point.

Local commands, which never touch Sail: ``status``.
Commands that create boxes or call the model: ``run``, ``judge``.
Control box commands: ``deploy``, ``launch``, ``remote-status``, ``fetch``.

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

from autoresearch import history
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


def cmd_run(args: argparse.Namespace) -> int:
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.model.sail_model import SailChatModel
    from autoresearch.orchestrator import run as run_mod
    from autoresearch.orchestrator.round import profile_docs_from_repo
    from autoresearch.worker.agent_loop import AgentLoopWorker

    cfg = _load(args.config)
    root = Path(args.repo_root).resolve()
    run_dir = Path(args.runs_root) / cfg.run_id if args.runs_root else cfg.run_dir
    docs = profile_docs_from_repo(root, cfg)
    paths = run_mod.init_run(cfg, run_dir, docs)
    boxes = SailBoxFactory(cfg)
    worker = AgentLoopWorker(SailChatModel(cfg.worker), boxes, cfg)
    log = paths.root / "run.log"
    last = run_mod.run(cfg, paths, worker, boxes, until_round=args.until, log=log)
    print(f"completed through round {last} of {cfg.rounds} in {paths.root}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Phase 2b check 1: judge hand written patches on one real referee box."""
    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.referee.referee import Referee

    cfg = _load(args.config)
    boxes = SailBoxFactory(cfg)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    box = boxes.create(name=f"judge-{ts}", role="referee")
    print(f"referee box {box.name} ({box.box_id})", flush=True)
    out_dir = Path(args.out) / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []
    try:
        ref = Referee(box, cfg)
        ref.setup()
        for patch_path in args.patches:
            patch = Path(patch_path).read_text()
            print(f"judging {patch_path} ...", flush=True)
            result = ref.judge(cfg.target.sha, patch)
            entry = {"patch": patch_path, "result": result.to_dict()}
            report.append(entry)
            print(
                f"  {result.verdict}: {result.reason} "
                f"(median {result.median_ratio}, ir {None if result.ir is None else result.ir.delta_pct:.2f}"
                f" pct, {result.wall_s:.0f}s)",
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

    p = sub.add_parser("judge", help="judge patch files on one real referee box")
    p.add_argument("config")
    p.add_argument("patches", nargs="+")
    p.add_argument("--out", default="runs", help="where the report is written")
    p.add_argument("--keep", action="store_true", help="leave the box running")
    p.set_defaults(func=cmd_judge)

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
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
