"""Live tracing of worker attempts, so a run can be watched while it happens.

The run directory is the record. This is a view and nothing else: no result,
plot or write up may read a number from here, and if the two ever disagree the
run directory is right.

The problem this solves: an attempt's transcript lives in memory until the
attempt returns, and with width above one the orchestrator holds every
transcript until the slowest worker of the round finishes. An attempt that runs
for fourteen minutes was invisible for all of it. A tracer is called at the same
four moments the transcript is appended to, so those moments leave the process
as they happen.

Nothing here may break a run. Every call into the platform is wrapped: a failure
is counted and reported once at the end of the attempt, never raised. A call
that is slow rather than failing is bounded too, because slow is the shape the
failure actually took: see ``FLUSH_TIMEOUT_S``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

from autoresearch.config import ObserveConfig, RunConfig
from autoresearch.model.protocol import Message, ModelResponse
from autoresearch.types import AttemptRef, Prediction, StopReason

REQUIRED_KEYS = ("SAIL_API_KEY",)

# How long an attempt will wait on a blocking call to the platform before giving
# up on it: starting the Voyage, and delivering its terminal event. In t1_w4b
# three of twelve workers lost about forty seconds each to a blocking ``flush``:
# the exporter met a five second read timeout and retried with backoff, and
# because a slow call raises nothing, the try/except around it never fired. 121
# seconds of experiment for zero traces. The Voyage SDK bounds each of these at
# ten seconds, which still counts against the worker's own clock.
FLUSH_TIMEOUT_S = 5.0

# Text in an event is clipped to this, the same length the on disk transcript
# keeps. The SDK replaces a payload over 64 KiB with a stub, so a whole file read
# back by a tool would otherwise arrive as nothing.
CLIP = 4000
ARGUMENTS_CLIP = 1000


class AttemptTrace(Protocol):
    """One worker attempt. Created when the attempt starts, ended once."""

    def box(self, box_id: str) -> None: ...

    def model_turn(self, turn: int, messages: list[Message], response: ModelResponse) -> None: ...

    def tool(self, turn: int, name: str, args: dict[str, Any], result: str) -> None: ...

    def end(
        self,
        stop: StopReason,
        *,
        patch: str | None = None,
        prediction: Prediction | None = None,
        error: str = "",
    ) -> None: ...


class Tracer(Protocol):
    def attempt(self, ref: AttemptRef, run_id: str, base_sha: str) -> AttemptTrace: ...


# ----- the default: nothing ------------------------------------------------------------


@dataclass(frozen=True)
class NullAttempt:
    def box(self, box_id: str) -> None: ...

    def model_turn(self, turn: int, messages: list[Message], response: ModelResponse) -> None: ...

    def tool(self, turn: int, name: str, args: dict[str, Any], result: str) -> None: ...

    def end(
        self,
        stop: StopReason,
        *,
        patch: str | None = None,
        prediction: Prediction | None = None,
        error: str = "",
    ) -> None: ...


@dataclass(frozen=True)
class NullTracer:
    """Used whenever tracing is off, unconfigured, or could not start."""

    def attempt(self, ref: AttemptRef, run_id: str, base_sha: str) -> AttemptTrace:  # noqa: ARG002
        return NullAttempt()


# ----- the hard boundary ---------------------------------------------------------------


@dataclass(frozen=True)
class SafeAttempt:
    """Swallows anything a tracer throws.

    The implementations below already guard themselves. This exists so the
    guarantee holds for any tracer at all, including ones written later: an
    attempt costs a worker box, a model call and up to an hour, and no view of
    it may ever be able to end it.
    """

    _inner: AttemptTrace

    def box(self, box_id: str) -> None:
        with suppress(Exception):
            self._inner.box(box_id)

    def model_turn(self, turn: int, messages: list[Message], response: ModelResponse) -> None:
        with suppress(Exception):
            self._inner.model_turn(turn, messages, response)

    def tool(self, turn: int, name: str, args: dict[str, Any], result: str) -> None:
        with suppress(Exception):
            self._inner.tool(turn, name, args, result)

    def end(
        self,
        stop: StopReason,
        *,
        patch: str | None = None,
        prediction: Prediction | None = None,
        error: str = "",
    ) -> None:
        with suppress(Exception):
            self._inner.end(stop, patch=patch, prediction=prediction, error=error)


def start_attempt(tracer: Tracer, ref: AttemptRef, run_id: str, base_sha: str) -> AttemptTrace:
    """The only way the worker starts a trace. Cannot raise."""
    try:
        return SafeAttempt(tracer.attempt(ref, run_id, base_sha))
    except Exception:
        return NullAttempt()


def _clip(s: str, n: int = CLIP) -> str:
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more]"


def _bounded(
    what: str,
    fn: Callable[[], Any],
    timeout: float,
    gave_up: threading.Event | None = None,
) -> tuple[Any, str]:
    """Run ``fn`` and return its value and ``""``, or ``None`` and why not.

    It also gives up on a call that is merely slow. The thread is left running
    as a daemon rather than killed, since there is no safe way to interrupt a
    socket read inside the SDK. It finishes or it dies with the process; either
    way the attempt has already moved on, and ``gave_up`` is set so the call can
    tell, when it does finish, that nobody is waiting for its result. The reason
    is printed, because the trace it would annotate is the one that failed, so
    the run's launch log is the only place it can be seen.
    """
    done: list[Any] = []
    caught: list[str] = []

    def run() -> None:
        try:
            done.append(fn())
        except Exception as e:
            caught.append(f"{what}: {e!r}")

    t = threading.Thread(target=run, name=f"voyage-{what}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        if gave_up is not None:
            gave_up.set()
        failure = f"{what}: still running after {timeout:.0f}s, abandoned"
    elif caught:
        failure = caught[0]
    else:
        return done[0], ""
    print(f"tracing: {failure}", flush=True)
    return None, failure


# ----- Sail Voyages --------------------------------------------------------------------


@dataclass
class VoyageAttempt:
    """One Voyage, recorded as events through its handle.

    Only explicit handle calls are used. Sail inference and box commands are not
    attributed to the Voyage: the tracer disables the SDK's current Voyage as
    soon as it is created, pins the worker's thread to that disabled state, and
    turns the SDK's automatic spans off; see ``VoyageTracer`` and
    ``build_tracer``. What the model was asked and answered is in ``model.turn``.

    Turn n's message count is only what was added since turn n-1: the tool
    results arrive as their own events, so the history is never resent.
    """

    _voyage: Any
    _agent: Any = None
    _sent: int = 0
    _failures: list[str] = field(default_factory=list)

    def _try(self, what: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            self._failures.append(f"{what}: {e!r}")

    def _event(self, kind: str, payload: dict[str, Any]) -> None:
        self._try(kind, lambda: self._voyage.event(kind, payload=payload))

    def box(self, box_id: str) -> None:
        self._event("box.ready", {"box_id": box_id})

    def model_turn(self, turn: int, messages: list[Message], response: ModelResponse) -> None:
        new = len(messages) - self._sent
        self._sent = len(messages)
        usage = response.usage
        self._event(
            "model.turn",
            {
                "turn": turn,
                "new_messages": new,
                "latency_s": round(response.latency_s, 3),
                "finish_reason": response.finish_reason,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "cached_tokens": usage.cached_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "content": _clip(response.content),
                "reasoning": _clip(response.reasoning),
                "tool_calls": [
                    {
                        "name": c.name,
                        "arguments": _clip(json.dumps(c.arguments, default=str), ARGUMENTS_CLIP),
                    }
                    for c in response.tool_calls
                ],
            },
        )

    def tool(self, turn: int, name: str, args: dict[str, Any], result: str) -> None:
        self._event(
            "tool.result",
            {
                "turn": turn,
                "name": name,
                "arguments": _clip(json.dumps(args, default=str), ARGUMENTS_CLIP),
                "result": _clip(result),
                "result_chars": len(result),
            },
        )

    def end(
        self,
        stop: StopReason,
        *,
        patch: str | None = None,
        prediction: Prediction | None = None,
        error: str = "",
    ) -> None:
        # The agent context was entered when the attempt started and must be
        # left on the same thread that entered it, which is this one.
        if self._agent is not None:
            agent, self._agent = self._agent, None
            self._try("end agent", lambda: agent.__exit__(None, None, None))
        output: dict[str, Any] = {
            "stop_reason": str(stop),
            "has_patch": bool(patch),
            "patch_bytes": len(patch or ""),
            "predicted_speedup": None if prediction is None else prediction.speedup,
            "error": _clip(error),
        }
        if self._failures:
            output["tracing_failures"] = list(self._failures)
        voyage = self._voyage
        if error:

            def send() -> None:
                voyage.fail(error_type=str(stop), message=_clip(error), payload=output)
        else:

            def send() -> None:
                voyage.complete(message=str(stop), payload=output)

        # Bounded, because the terminal call flushes and that is the call that
        # blocked three workers in t1_w4b: a trace may be lost, an attempt may
        # not be held up.
        _, failure = _bounded("end", send, FLUSH_TIMEOUT_S)
        if failure:
            self._failures.append(failure)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(self._failures)


@dataclass
class VoyageTracer:
    """One per run. Each attempt is its own Voyage in the run's series.

    ``_sdk`` is the ``sail.voyage`` module, a field so tests can stand in for it.
    """

    _sdk: Any
    _series: str
    _model: str

    def attempt(self, ref: AttemptRef, run_id: str, base_sha: str) -> AttemptTrace:
        sdk = self._sdk
        metadata = {
            "run_id": run_id,
            "attempt": ref.dirname,
            "number": ref.number,
            "round": ref.round,
            "worker": ref.worker,
            "base_sha": base_sha,
            "model": self._model,
        }
        # The SDK resolves a thread's Voyage from its own context first and the
        # process wide fallback second. A worker thread never creates in its own
        # context (``create`` runs on the bounded thread below), so without this
        # its model calls would resolve the fallback, which between another
        # attempt's ``create`` and ``disable`` is that other attempt's Voyage.
        # Pinning this thread to the disabled state keeps its model calls out of
        # every Voyage, whatever the fallback holds at the time.
        with suppress(Exception):
            sdk.disable()
        gave_up = threading.Event()

        def start() -> Any:
            voyage = sdk.create(self._series, metadata=metadata)
            # ``create`` also makes this the process wide fallback Voyage, which
            # every thread without its own resolves to: the box threads of this
            # worker, other workers and the referee. Reset at once, in the same
            # thread, so nothing is attributed by accident, even when this runs
            # after the bound has given up on it.
            sdk.disable()
            if gave_up.is_set():
                # Nobody will end this Voyage, so end it here rather than leave
                # it running on the dashboard for good.
                with suppress(Exception):
                    voyage.fail(
                        error_type="tracing_abandoned",
                        message=f"the attempt did not wait {FLUSH_TIMEOUT_S:.0f}s for this"
                        " Voyage to start; it ran untraced",
                    )
            return voyage

        voyage, _ = _bounded(f"start {ref.dirname}", start, FLUSH_TIMEOUT_S, gave_up)
        if voyage is None or voyage.id is None:
            return NullAttempt()
        trace = VoyageAttempt(_voyage=voyage)
        try:
            agent = voyage.agent("Worker", role="worker")
            agent.__enter__()
            trace._agent = agent
        except Exception as e:
            trace._failures.append(f"agent: {e!r}")
        print(f"tracing: attempt {ref.dirname} {voyage.dashboard_url}", flush=True)
        return trace


def build_tracer(config: RunConfig) -> Tracer:
    """The tracer named by the config, or ``NullTracer`` with a reason on stdout.

    Tracing that cannot start is never fatal: the experiment runs regardless.
    """
    observe = config.observe
    if not observe.enabled:
        return NullTracer()
    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        print(f"tracing off: {' and '.join(missing)} not set", flush=True)
        return NullTracer()
    try:
        from sail import voyage as sdk
    except Exception as e:
        print(f"tracing off: sail.voyage failed to import: {e!r}", flush=True)
        return NullTracer()
    # The SDK synthesises a span for any box command or model call made while a
    # Voyage is current, and a fresh thread (every box command runs on one) is
    # current on the process wide fallback. That fallback is an attempt's Voyage
    # for the moment between its ``create`` and ``disable``, so with the switch
    # on, another worker's box command in that moment would land in its trace.
    # Off, and only the events this module emits through a handle are recorded.
    os.environ.setdefault("SAIL_VOYAGE_AUTO_SPANS", "0")
    series = session_id(observe, config.run_id)
    print(f"tracing on: voyage series {series}", flush=True)
    return VoyageTracer(_sdk=sdk, _series=series, _model=config.worker.model)


def session_id(observe: ObserveConfig, run_id: str) -> str:
    return f"{observe.session_prefix}-{run_id}" if observe.session_prefix else run_id
