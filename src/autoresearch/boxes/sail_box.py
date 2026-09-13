"""The only module that talks to the Sail SDK for sailboxes.

``SailBox`` adapts one ``sail.Sailbox`` to the ``Box`` protocol. ``SailBoxFactory``
creates boxes for a role from the run's image and sizes. Nothing here is
exercised by unit tests; it is exercised by the Phase 2b checks.

Every call into the SDK carries a host side deadline. The SDK's own ``timeout``
bounds the command running inside the guest, not the call: "when timeout
elapses, the command is killed". If the box itself has gone there is no command
left to kill, so nothing brings the call back. On 2026-09-12 that stopped a run
dead: all four worker boxes went from running to terminated with no error
message, and all four ``box.run`` calls blocked in the SDK's
``ExecProcessHandle.wait_stream_ended`` for as long as the process lived. No
exception, no log line, no round. The orchestrator waited on them for half an
hour and would have waited forever.

Passing a deadline raises ``BoxError``, which every caller already handles: a
worker records ``BOX_ERROR`` and the round goes on, the referee pool rebuilds
the box. One unreachable box then costs one attempt instead of the whole run.
"""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.boxes.protocol import Box, BoxError, CommandResult
from autoresearch.config import RunConfig

ROLE_SIZES = {"worker": "worker_size", "referee": "referee_size", "control": "control_size"}

# Slack on top of the timeout a command was already given, so the SDK's own
# timeout is what fires when the command is merely slow, and this guard fires
# only when the SDK has stopped answering at all.
RUN_GRACE_S = 120
# For the calls that carry no timeout of their own: file transfers, terminate.
SDK_DEADLINE_S = 300
APP_NAME = "autoresearch"
CREATE_TIMEOUT_S = 1800
# How far before a failed create a box of the same name may be stamped and still
# count as the one that create brought up. Covers clock skew between host and Sail.
CLOCK_SLACK = dt.timedelta(minutes=5)


def _guarded[T](what: str, deadline: float, call: Callable[[], T]) -> T:
    """Run one SDK call with a wall clock deadline, as BoxError on failure or overrun.

    The thread is a daemon and is abandoned rather than joined when the deadline
    passes, because a thread parked inside the SDK cannot be interrupted and a
    non daemon one would hang interpreter exit as well as this caller. It leaks
    until the process ends, which is the price of getting the caller back.
    """
    done: list[T] = []
    failed: list[Exception] = []

    def target() -> None:
        try:
            done.append(call())
        except Exception as e:
            failed.append(e)

    thread = threading.Thread(target=target, name=f"sail-{what}", daemon=True)
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        raise BoxError(f"{what} did not return within {deadline:.0f}s; the box is unreachable")
    if failed:
        raise BoxError(f"{what} failed: {failed[0]!r}") from failed[0]
    return done[0]


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
        r = _guarded(
            f"run on {self.name}",
            timeout + RUN_GRACE_S,
            lambda: self._sb.run(command, timeout=timeout, env=dict(env) if env else None, cwd=cwd),
        )
        return CommandResult(
            exit_code=int(r.exit_code),
            stdout=str(r.stdout or ""),
            stderr=str(r.stderr or ""),
            timed_out=bool(r.timed_out),
        )

    def start(self, command: str, *, env: Mapping[str, str] | None = None) -> None:
        _guarded(
            f"start on {self.name}",
            SDK_DEADLINE_S,
            lambda: self._sb.exec(command, background=True, env=dict(env) if env else None).wait(),
        )

    def write(self, path: str, data: bytes) -> None:
        _guarded(
            f"write {path} on {self.name}", SDK_DEADLINE_S, lambda: self._sb.fs.write(path, data)
        )

    def read(self, path: str) -> bytes:
        return bytes(
            _guarded(f"read {path} on {self.name}", SDK_DEADLINE_S, lambda: self._sb.fs.read(path))
        )

    def upload_dir(self, local: Path, remote: str) -> None:
        _guarded(
            f"upload {local} on {self.name}",
            SDK_DEADLINE_S,
            lambda: self._sb.fs.upload_dir(str(local), remote),
        )

    def download_dir(self, remote: str, local: Path) -> None:
        _guarded(
            f"download {remote} on {self.name}",
            SDK_DEADLINE_S,
            lambda: self._sb.fs.download_dir(remote, str(local)),
        )

    def terminate(self) -> None:
        # Never swallowed here: a box that fails to terminate bills until
        # autosleep, and the caller must know. Terminate is idempotent on the
        # Sail side. The worker is the one caller that catches it, because by
        # then its attempt is already finished and worth more than the teardown.
        _guarded(f"terminate {self.name}", SDK_DEADLINE_S, lambda: self._sb.terminate())


@dataclass(frozen=True)
class LiveBox:
    """A box in the app that is not terminated, as ``reap`` lists it."""

    box_id: str
    name: str
    status: str
    created_at: str


def live_boxes(app_name: str = APP_NAME, prefix: str = "") -> list[LiveBox]:
    """Every box in the app that is not terminated, oldest first, optionally by name prefix."""
    import sail

    app = sail.App.find(app_name, mint_if_missing=True)
    rows = _guarded("list boxes", SDK_DEADLINE_S, lambda: sail.Sailbox.list(app_id=app))
    live = [
        LiveBox(
            box_id=str(sb.sailbox_id),
            name=str(sb.name),
            status=str(sb.status),
            created_at=str(getattr(sb, "created_at", "")),
        )
        for sb in rows
        if str(sb.status) != "terminated" and str(sb.name).startswith(prefix)
    ]
    return sorted(live, key=lambda b: b.created_at)


def terminate_box(box_id: str) -> None:
    import sail

    _guarded(f"terminate {box_id}", SDK_DEADLINE_S, lambda: sail.Sailbox.get(box_id).terminate())


class SailBoxFactory:
    """Creates boxes from the run's image. One instance per orchestrator process."""

    def __init__(self, config: RunConfig, app_name: str = APP_NAME) -> None:
        import sail

        from autoresearch.boxes.image import build_image

        self._sail = sail
        self._config = config
        self._app = sail.App.find(app_name, mint_if_missing=True)
        self._image = build_image(config.target)

    def _create(self, name: str, what: str, **kwargs: Any) -> Box:
        """Create a box that never sleeps, and clean up after a create that raised."""
        since = dt.datetime.now(dt.UTC)
        try:
            sb = self._sail.Sailbox.create(
                app=self._app,
                name=name,
                image=self._image,
                auto_sleep=self._sail.AutoSleep.never(),
                timeout=CREATE_TIMEOUT_S,
                **kwargs,
            )
        except Exception as e:
            leftover = self._end_late_arrival(name, since)
            raise BoxError(f"create {what} failed: {e!r}{leftover}") from e
        return SailBox(sb)

    def _end_late_arrival(self, name: str, since: dt.datetime) -> str:
        """Terminate a box that came up although its create raised. A suffix for the error.

        The SDK says a create that times out raises "and the Sailbox may still come
        up in the background". Nothing would hold a handle to that box, and it
        never sleeps. This finds one that is up by the time the error is handled,
        by exact name among boxes created since the call. One still queued is not
        seen yet, which is what ``autoresearch reap`` is for.
        """
        reap = "; check with autoresearch reap"
        try:
            found = _guarded(
                f"list {name}",
                SDK_DEADLINE_S,
                lambda: self._sail.Sailbox.list(app_id=self._app, search=name),
            )
        except BoxError as e:
            return f"; could not look for a box that came up anyway ({e}){reap}"
        ended: list[str] = []
        for sb in found:
            if str(sb.name) != name or str(sb.status) == "terminated":
                continue
            created = getattr(sb, "created_at", None)
            if (
                isinstance(created, dt.datetime)
                and created.tzinfo is not None
                and created < since - CLOCK_SLACK
            ):
                continue
            try:
                _guarded(f"terminate {name}", SDK_DEADLINE_S, sb.terminate)
            except BoxError as e:
                return f"; {sb.sailbox_id} came up anyway and did not terminate ({e}){reap}"
            ended.append(str(sb.sailbox_id))
        return f"; terminated {', '.join(ended)}, which came up anyway" if ended else ""

    def create(self, *, name: str, role: str) -> Box:
        """A worker or referee box. Never sleeps, for the same reason the control box does not.

        A worker box is idle by the definition autosleep uses for most of an
        attempt: the model is thinking and nothing is running in the guest. In
        the first t1_w4c attempt all four worker boxes checkpointed twice and
        were then terminated, and the four ``box.run`` calls in flight never
        returned. Waking is meant to be transparent, so sleeping may not be the
        whole cause, but an idle box is the one condition all four shared and
        there is nothing to gain by letting a box in the middle of an attempt
        sleep. The wall clock guard in ``SailBox.run`` is the other half of this.
        """
        size = getattr(self._config.boxes, ROLE_SIZES[role])
        return self._create(
            name,
            f"{name} ({role}, size {size})",
            size=size,
            disk_limit_gib=self._config.boxes.disk_gib,
        )

    def create_control(self, *, name: str, volume: str, mount: str) -> Box:
        """The control box: never sleeps, with the run volume mounted at ``mount``."""
        try:
            vol = self._sail.Volume.find(volume, mint_if_missing=True)
        except Exception as e:
            raise BoxError(f"create control box {name} failed: {e!r}") from e
        return self._create(
            name,
            f"control box {name}",
            size=self._config.boxes.control_size,
            disk_limit_gib=self._config.boxes.disk_gib,
            volumes={mount: vol},
        )

    def reattach(self, box_id: str) -> Box | None:
        try:
            sb = self._sail.Sailbox.get(box_id)
        except Exception:
            return None
        if str(getattr(sb, "status", "")) not in ("running", "sleeping", "paused"):
            return None
        return SailBox(sb)
