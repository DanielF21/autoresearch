"""CLI registration for the control box commands. Imported by ``cli.build_parser``."""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

from autoresearch import env
from autoresearch.config import load_config
from autoresearch.control import deploy as ctl

GONE = (
    "control box {box_id} is not running. It terminates itself when the last run on it "
    "ends; fetch still works, and deploy brings up a new one."
)


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
        raise SystemExit(GONE.format(box_id=info.box_id))
    return cfg, box


def cmd_launch(args: argparse.Namespace) -> int:
    cfg, box = _control_box(args.config)
    envs = _launch_env()
    cmd = ctl.launch(
        cfg,  # type: ignore[arg-type]
        args.config,
        box,  # type: ignore[arg-type]
        envs,
        args.until,
        keep_control=args.keep_control,
    )
    print(f"started in the control box: {cmd}")
    traced = load_config(Path(args.config)).observe.enabled
    print("tracing: " + ("on, as Sail Voyages" if traced else "off in this config"))
    if args.keep_control:
        print("the control box stays up after the run: terminate it yourself, or autoresearch reap")
    else:
        print("the control box terminates itself once no run is going on it")
    print("read progress with: autoresearch remote-status " + args.config)
    return 0


def cmd_remote_status(args: argparse.Namespace) -> int:
    cfg, box = _control_box(args.config)
    print(ctl.remote_status(cfg, box))  # type: ignore[arg-type]
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Copy a run down. Through the control box if it is up, else a temporary box on the volume."""
    from autoresearch.boxes.sail_box import SailBoxFactory

    env.load_dotenv()
    cfg = load_config(Path(args.config))
    info = ctl.load_control()
    factory = SailBoxFactory(cfg)
    box = factory.reattach(info.box_id)
    temporary = box is None
    if box is None:
        name = f"fetch-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"control box is gone; bringing up {name} on the volume to fetch", flush=True)
        box = factory.create_control(name=name, volume=cfg.storage.volume, mount=cfg.storage.mount)
    try:
        local = ctl.fetch(cfg, box, Path(args.dest))
    finally:
        if temporary:
            box.terminate()
            print(f"terminated {box.name}", flush=True)
    print(f"fetched to {local}")
    return 0


def cmd_release_control(args: argparse.Namespace) -> int:
    """Terminate this control box unless another run is still going on it. Run by a launch."""
    others = ctl.other_runs()
    if others:
        print(f"control box kept: runs still going, pids {list(others)}", flush=True)
        return 0
    from autoresearch.boxes.sail_box import SailBoxFactory

    cfg = load_config(Path(args.config))
    box = SailBoxFactory(cfg).reattach(args.box_id)
    if box is None:
        print(f"control box {args.box_id} is already gone", flush=True)
        return 0
    print(f"no run is going; terminating control box {args.box_id}", flush=True)
    box.terminate()
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
    p.add_argument(
        "--keep-control",
        action="store_true",
        help="leave the control box running after the run ends (it never sleeps)",
    )
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("remote-status", help="status of a run, read from the control box")
    p.add_argument("config")
    p.set_defaults(func=cmd_remote_status)

    p = sub.add_parser("fetch", help="copy a run directory from the volume to the laptop")
    p.add_argument("config")
    p.add_argument("--dest", default="runs")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser(
        "release-control",
        help="terminate the control box if no run is going on it (a launch runs this)",
    )
    p.add_argument("config")
    p.add_argument("box_id")
    p.set_defaults(func=cmd_release_control)
