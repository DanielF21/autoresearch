"""The narrow view of a sailbox the harness needs.

Everything the referee and the worker do to a box goes through this protocol:
run a command, move files, terminate. That is what makes them testable against
``FakeBox`` and what keeps the Sail SDK confined to one module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class BoxError(RuntimeError):
    """A box operation failed at the platform level, not inside the command."""


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def last_json_line(self) -> str | None:
        """Guest programs print one JSON object as their last line."""
        for line in reversed(self.stdout.splitlines()):
            if line.startswith("{"):
                return line
        return None


class Box(Protocol):
    @property
    def box_id(self) -> str: ...

    @property
    def name(self) -> str: ...

    def run(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> CommandResult: ...

    def start(self, command: str, *, env: Mapping[str, str] | None = None) -> None:
        """Start a detached process and return at once. Output is the command's business."""
        ...

    def write(self, path: str, data: bytes) -> None: ...

    def read(self, path: str) -> bytes: ...

    def upload_dir(self, local: Path, remote: str) -> None: ...

    def download_dir(self, remote: str, local: Path) -> None: ...

    def terminate(self) -> None: ...


class BoxFactory(Protocol):
    """Creates boxes for a role. The role names the image and size to use."""

    def create(self, *, name: str, role: str) -> Box: ...

    def create_control(self, *, name: str, volume: str, mount: str) -> Box:
        """The long lived control box: never sleeps, with the run volume mounted."""
        ...

    def reattach(self, box_id: str) -> Box | None:
        """A live box by id, or None if it is gone. Used on resume."""
        ...
