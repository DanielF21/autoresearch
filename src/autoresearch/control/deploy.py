"""The control box: where the orchestrator runs while the laptop is off.

``deploy`` creates it once with the run volume mounted, uploads this package
and installs it. ``launch`` starts one run as a detached process with the API
key in that process's environment only. ``remote_status`` runs the status
command inside the box. ``fetch`` pulls a run directory down to the laptop.

The box id is kept in ``runs/control.json`` on the laptop so later commands
find it. Nothing here needs the key except ``launch``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autoresearch.boxes.protocol import Box, BoxError, BoxFactory
from autoresearch.config import RunConfig

PACKAGE_DIR = "/workspace/autoresearch"
CONTROL_RECORD = Path("runs") / "control.json"

# Sail's interpreter installs console scripts to a directory that is not on the
# box's PATH, so `autoresearch ...` is not callable there. Everything inside a
# box goes through the same interpreter that does the install: it is resolved
# once per command, so no state has to be carried between commands, and the
# package is guaranteed importable by whatever installed it.
PYTHON = '"$(command -v python3 || command -v python)"'
CLI = f"{PYTHON} -m autoresearch"
UPLOAD_IGNORE = (
    ".git",
    ".venv",
    "runs",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)


@dataclass(frozen=True)
class ControlInfo:
    box_id: str
    name: str
    volume: str
    mount: str

    def to_dict(self) -> dict[str, str]:
        return {
            "box_id": self.box_id,
            "name": self.name,
            "volume": self.volume,
            "mount": self.mount,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> ControlInfo:
        return cls(box_id=d["box_id"], name=d["name"], volume=d["volume"], mount=d["mount"])


def save_control(info: ControlInfo, record: Path = CONTROL_RECORD) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(info.to_dict(), indent=2) + "\n")


def load_control(record: Path = CONTROL_RECORD) -> ControlInfo:
    if not record.exists():
        raise BoxError(f"no control box recorded at {record}; run deploy first")
    return ControlInfo.from_dict(json.loads(record.read_text()))


def _staging_copy(repo_root: Path, staging: Path) -> None:
    """Copy the package source without the directories that must not go up."""
    import shutil

    shutil.copytree(
        repo_root,
        staging,
        ignore=shutil.ignore_patterns(*UPLOAD_IGNORE),
        dirs_exist_ok=True,
    )


def deploy(
    config: RunConfig, boxes: BoxFactory, repo_root: Path, staging: Path, name: str
) -> ControlInfo:
    """Create the control box, upload the package, install it. Returns its record."""
    box = boxes.create_control(name=name, volume=config.storage.volume, mount=config.storage.mount)
    try:
        _staging_copy(repo_root, staging)
        box.upload_dir(staging, PACKAGE_DIR)
        r = box.run(
            f"cd {PACKAGE_DIR} && {PYTHON} -m pip install -q -e . "
            f"&& {CLI} --help >/dev/null && echo installed",
            timeout=900,
        )
        if not r.ok or "installed" not in r.stdout:
            raise BoxError(f"package install failed on the control box: {r.stderr[-800:]}")
        r = box.run(
            f"mkdir -p {config.storage.mount}/runs && ls {config.storage.mount}", timeout=60
        )
        if not r.ok:
            raise BoxError(f"volume not mounted at {config.storage.mount}: {r.stderr[-300:]}")
    except BoxError:
        box.terminate()
        raise
    return ControlInfo(
        box_id=box.box_id, name=box.name, volume=config.storage.volume, mount=config.storage.mount
    )


def launch_command(config: RunConfig, config_path: str, until: int | None = None) -> str:
    """The detached command that runs one setting inside the control box."""
    log = f"{config.storage.mount}/runs/{config.run_id}.launch.log"
    stop = f" --until {until}" if until is not None else ""
    return (
        f"cd {PACKAGE_DIR} && nohup {CLI} run {config_path} --repo-root {PACKAGE_DIR}"
        f"{stop} >> {log} 2>&1 &"
    )


def launch(
    config: RunConfig, config_path: str, box: Box, api_key: str, until: int | None = None
) -> str:
    """Start the run. The key lives only in this process's environment."""
    cmd = launch_command(config, config_path, until)
    box.start(cmd, env={"SAIL_API_KEY": api_key})
    return cmd


def remote_status(config: RunConfig, box: Box) -> str:
    run_dir = f"{config.storage.mount}/runs/{config.run_id}"
    r = box.run(f"cd {PACKAGE_DIR} && {CLI} status {run_dir}", timeout=120)
    if not r.ok:
        return f"status failed (rc {r.exit_code}): {r.stderr[-800:] or r.stdout[-800:]}"
    return r.stdout


def fetch(config: RunConfig, box: Box, dest: Path) -> Path:
    """Copy the run directory from the volume to ``dest/<run_id>``."""
    run_dir = f"{config.storage.mount}/runs/{config.run_id}"
    local = dest / config.run_id
    local.mkdir(parents=True, exist_ok=True)
    box.download_dir(run_dir, local)
    return local
