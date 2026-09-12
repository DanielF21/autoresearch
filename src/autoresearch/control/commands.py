"""CLI registration for the control box commands. Imported by ``cli.build_parser``."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from pathlib import Path

from autoresearch.config import load_config
from autoresearch.control import deploy as ctl


def _factory(config_path: str) -> tuple[object, object]:
    from autoresearch.boxes.sail_box import SailBoxFactory

    cfg = load_config(Path(config_path))
    return cfg, SailBoxFactory(cfg)


def _api_key() -> str:
    key = os.environ.get("SAIL_API_KEY", "")
    if not key:
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("SAIL_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        raise SystemExit("SAIL_API_KEY is not set and .env has no SAIL_API_KEY line")
    return key


def cmd_deploy(args: argparse.Namespace) -> int:
    from autoresearch.boxes.sail_box import SailBoxFactory

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

    cfg = load_config(Path(config_path))
    info = ctl.load_control()
    box = SailBoxFactory(cfg).reattach(info.box_id)
    if box is None:
        raise SystemExit(f"control box {info.box_id} is not running; deploy again")
    return cfg, box


def cmd_launch(args: argparse.Namespace) -> int:
    cfg, box = _control_box(args.config)
    cmd = ctl.launch(cfg, args.config, box, _api_key())  # type: ignore[arg-type]
    print(f"started in the control box: {cmd}")
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
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("remote-status", help="status of a run, read from the control box")
    p.add_argument("config")
    p.set_defaults(func=cmd_remote_status)

    p = sub.add_parser("fetch", help="copy a run directory from the volume to the laptop")
    p.add_argument("config")
    p.add_argument("--dest", default="runs")
    p.set_defaults(func=cmd_fetch)
