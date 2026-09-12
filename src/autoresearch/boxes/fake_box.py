"""An in memory box for tests.

Files live in a dict. Commands are answered by handlers registered with
``on(substring, result)`` or ``on(substring, callable)``; the first handler
whose substring appears in the command wins. Every command and every file
write is recorded so a test can assert exactly what the harness did.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.boxes.protocol import Box, BoxError, CommandResult

Handler = Callable[[str], CommandResult] | CommandResult


def ok(stdout: str = "", stderr: str = "") -> CommandResult:
    return CommandResult(0, stdout, stderr)


def fail(stderr: str = "boom", code: int = 1, stdout: str = "") -> CommandResult:
    return CommandResult(code, stdout, stderr)


@dataclass
class FakeBox:
    box_id: str = "sb_fake"
    name: str = "fake"
    files: dict[str, bytes] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    uploads: list[tuple[Path, str]] = field(default_factory=list)
    terminated: bool = False
    _handlers: list[tuple[str, Handler]] = field(default_factory=list)

    def on(self, substring: str, handler: Handler, *, first: bool = False) -> FakeBox:
        """Register a handler. ``first`` puts it ahead of earlier, broader matches."""
        if first:
            self._handlers.insert(0, (substring, handler))
        else:
            self._handlers.append((substring, handler))
        return self

    def run(
        self,
        command: str,
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        if self.terminated:
            raise BoxError(f"{self.name} is terminated")
        self.commands.append(command)
        for substring, handler in self._handlers:
            if substring in command:
                return handler(command) if callable(handler) else handler
        return CommandResult(127, "", f"fake box has no handler for: {command}")

    def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def read(self, path: str) -> bytes:
        if path not in self.files:
            raise BoxError(f"no such file in fake box: {path}")
        return self.files[path]

    def upload_dir(self, local: Path, remote: str) -> None:
        self.uploads.append((local, remote))
        for p in local.rglob("*"):
            if p.is_file():
                self.files[f"{remote}/{p.relative_to(local)}"] = p.read_bytes()

    def terminate(self) -> None:
        self.terminated = True


@dataclass
class FakeBoxFactory:
    """Hands out FakeBoxes and remembers them. ``prepare`` customises each new box."""

    prepare: Callable[[FakeBox, str], None] | None = None
    created: list[FakeBox] = field(default_factory=list)
    _counter: int = 0

    def create(self, *, name: str, role: str) -> Box:
        self._counter += 1
        box = FakeBox(box_id=f"sb_fake_{self._counter}", name=name)
        if self.prepare is not None:
            self.prepare(box, role)
        self.created.append(box)
        return box

    def reattach(self, box_id: str) -> Box | None:
        for box in self.created:
            if box.box_id == box_id and not box.terminated:
                return box
        return None
