"""CLI registration for the control box commands. Imported by ``cli.build_parser``."""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

from autoresearch import env
from autoresearch.config import load_config
from autoresearch.control import deploy as ctl


def _launch_env() -> dict[str, str]:
    return env.launch_env()


def cmd_deploy(args: argparse.Namespace) -> int:
    from autoresearch.boxes.sail_box import SailBoxFactory

    env.load_dotenv()
    cfg = load_config(Path(args.config))
    boxes = SailBoxFactory(cfg)
    name = f"control-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with tempfile.TemporaryDirectory() as tmp:
        info = ctl.deploy(cfg, boxes, Path(args.repo_root).resolve(), Path(tmp) / "pkg", name)
    ctl.save_control(info)
    print(f"control box {info.name} ({info.box_id}) with volume {info.volume} at {info.mount}")
    print(f"recorded in {ctl.CONTROL_RECORD}")
    return 0


def _control_box(config_path: str) -> tuple[object, object]:
    from autoresearch.boxes.sail_box import SailBoxFactory

    env.load_dotenv()
    cfg = load_config(Path(config_path))
    info = ctl.load_control()
    box = SailBoxFactory(cfg).reattach(info.box_id)
    if box is None:
        raise SystemExit(f"control box {info.box_id} is not running; deploy again")
    return cfg, box


def cmd_launch(args: argparse.Namespace) -> int:
    cfg, box = _control_box(args.config)
    envs = _launch_env()
    cmd = ctl.launch(cfg, args.config, box, envs, args.until)  # type: ignore[arg-type]
    traced = [k for k in env.TRACING_KEYS if k in envs]
    print(f"started in the control box: {cmd}")
    print("tracing keys forwarded: " + (", ".join(traced) if traced else "none"))
    print("read progress with: autoresearch remote-status " + args.config)
    return 0


def cmd_remote_status(args: argparse.Namespace) -> int:
    cfg, box = _control_box(args.config)
    print(ctl.remote_status(cfg, box))  # type: ignore[arg-type]
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    cfg, box = _control_box(args.config)
    local = ctl.fetch(cfg, box, Path(args.dest))  # type: ignore[arg-type]
    print(f"fetched to {local}")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "deploy", help="create the control box with the run volume and install the package"
    )
    p.add_argument("config")
    p.add_argument("--repo-root", default=".")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("launch", help="start one setting's run inside the control box")
    p.add_argument("config", help="config path relative to the repo root, as uploaded")
    p.add_argument("--until", type=int, default=None, help="stop after this round")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("remote-status", help="status of a run, read from the control box")
    p.add_argument("config")
    p.set_defaults(func=cmd_remote_status)

    p = sub.add_parser("fetch", help="copy a run directory from the volume to the laptop")
    p.add_argument("config")
    p.add_argument("--dest", default="runs")
    p.set_defaults(func=cmd_fetch)
