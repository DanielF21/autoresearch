"""SIGTERM and SIGHUP become an exit that runs ``finally``, so boxes get terminated."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Iterator

import pytest

from autoresearch import shutdown


@pytest.fixture(autouse=True)
def restore_handlers() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in shutdown.EXIT_SIGNALS}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_sigterm_and_sighup_become_a_clean_exit() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    shutdown.exit_cleanly_on_signals()
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler) and signal.getsignal(signal.SIGHUP) is handler
    with pytest.raises(SystemExit) as info:
        handler(signal.SIGTERM, None)
    assert info.value.code == 128 + signal.SIGTERM
    # A second signal cannot interrupt the teardown the first one started.
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN


def test_a_real_sigterm_unwinds_through_finally() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    shutdown.exit_cleanly_on_signals()
    cleaned: list[bool] = []
    with pytest.raises(SystemExit):
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # the handler raises out of here
        finally:
            cleaned.append(True)
    assert cleaned == [True]


def test_a_run_under_nohup_keeps_ignoring_sighup() -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    shutdown.exit_cleanly_on_signals()
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    assert callable(signal.getsignal(signal.SIGTERM))
