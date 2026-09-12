"""The only module that talks to the Sail SDK for sailboxes.

``SailBox`` adapts one ``sail.Sailbox`` to the ``Box`` protocol. ``SailBoxFactory``
creates boxes for a role from the run's image and sizes. Nothing here is
exercised by unit tests; it is exercised by the Phase 2b checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.boxes.protocol import Box, BoxError, CommandResult
from autoresearch.config import RunConfig

ROLE_SIZES = {"worker": "worker_size", "referee": "referee_size", "control": "control_size"}


@dataclass(frozen=True)
class SailBox:
    _sb: Any

    @property
    def box_id(self) -> str:
        return str(self._sb.sailbox_id)

    @property
    def name(self) -> str:
        return str(self._sb.name)

    def run(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        try:
            r = self._sb.run(command, timeout=timeout, env=dict(env) if env else None, cwd=cwd)
        except Exception as e:
            raise BoxError(f"run failed on {self.name}: {e!r}") from e
        return CommandResult(
            exit_code=int(r.exit_code),
            stdout=str(r.stdout or ""),
            stderr=str(r.stderr or ""),
            timed_out=bool(r.timed_out),
        )

    def start(self, command: str, *, env: Mapping[str, str] | None = None) -> None:
        try:
            self._sb.exec(command, background=True, env=dict(env) if env else None).wait()
        except Exception as e:
            raise BoxError(f"start failed on {self.name}: {e!r}") from e

    def write(self, path: str, data: bytes) -> None:
        try:
            self._sb.fs.write(path, data)
        except Exception as e:
            raise BoxError(f"write {path} failed on {self.name}: {e!r}") from e

    def read(self, path: str) -> bytes:
        try:
            return bytes(self._sb.fs.read(path))
        except Exception as e:
            raise BoxError(f"read {path} failed on {self.name}: {e!r}") from e

    def upload_dir(self, local: Path, remote: str) -> None:
        try:
            self._sb.fs.upload_dir(str(local), remote)
        except Exception as e:
            raise BoxError(f"upload {local} failed on {self.name}: {e!r}") from e

    def download_dir(self, remote: str, local: Path) -> None:
        try:
            self._sb.fs.download_dir(remote, str(local))
        except Exception as e:
            raise BoxError(f"download {remote} failed on {self.name}: {e!r}") from e

    def terminate(self) -> None:
        # Never swallowed: a box that fails to terminate bills until autosleep,
        # and the caller must know. Terminate is idempotent on the Sail side.
        self._sb.terminate()


class SailBoxFactory:
    """Creates boxes from the run's image. One instance per orchestrator process."""

    def __init__(self, config: RunConfig, app_name: str = "autoresearch") -> None:
        import sail

        from autoresearch.boxes.image import build_image

        self._sail = sail
        self._config = config
        self._app = sail.App.find(app_name, mint_if_missing=True)
        self._image = build_image(config.target)

    def create(self, *, name: str, role: str) -> Box:
        size = getattr(self._config.boxes, ROLE_SIZES[role])
        try:
            sb = self._sail.Sailbox.create(
                app=self._app,
                name=name,
                image=self._image,
                size=size,
                disk_limit_gib=self._config.boxes.disk_gib,
                timeout=1800,
            )
        except Exception as e:
            raise BoxError(f"create {name} ({role}, size {size}) failed: {e!r}") from e
        return SailBox(sb)

    def create_control(self, *, name: str, volume: str, mount: str) -> Box:
        """The control box: never sleeps, with the run volume mounted at ``mount``."""
        try:
            vol = self._sail.Volume.find(volume, mint_if_missing=True)
            sb = self._sail.Sailbox.create(
                app=self._app,
                name=name,
                image=self._image,
                size=self._config.boxes.control_size,
                disk_limit_gib=self._config.boxes.disk_gib,
                volumes={mount: vol},
                auto_sleep=self._sail.AutoSleep.never(),
                timeout=1800,
            )
        except Exception as e:
            raise BoxError(f"create control box {name} failed: {e!r}") from e
        return SailBox(sb)

    def reattach(self, box_id: str) -> Box | None:
        try:
            sb = self._sail.Sailbox.get(box_id)
        except Exception:
            return None
        if str(getattr(sb, "status", "")) not in ("running", "sleeping", "paused"):
            return None
        return SailBox(sb)
