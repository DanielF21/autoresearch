"""Turn the signals that would kill a process without cleanup into an exit that cleans up.

Python's default for SIGTERM and SIGHUP is to die at once. No ``finally`` runs,
so every sailbox the process holds is left running and billing. ``kill <pid>``
and closing the terminal a run was started from both send one of them. Raising
SystemExit from the handler instead unwinds the main thread through every
``finally``, which is where boxes are terminated.

After the first signal both are ignored, so a second one cannot interrupt the
teardown the first one started. SIGKILL is the way to stop without cleanup.

SIGHUP is left alone when it is already ignored, which is what ``nohup`` sets
up: a run started under nohup is meant to outlive its terminal.

Nothing in a process can survive SIGKILL, a dead laptop, or a lost control box.
``autoresearch reap`` is the backstop for those.
"""

from __future__ import annotations

import signal
import sys
from types import FrameType

EXIT_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _exit(signum: int, _frame: FrameType | None) -> None:
    for sig in EXIT_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    print(
        f"received {signal.Signals(signum).name}: finishing in flight attempts and terminating "
        "boxes before exit. SIGKILL stops without cleanup.",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(128 + signum)


def exit_cleanly_on_signals() -> None:
    """Install the handler for every exit signal not already ignored. Main thread only."""
    for sig in EXIT_SIGNALS:
        if signal.getsignal(sig) is signal.SIG_IGN:
            continue
        signal.signal(sig, _exit)
