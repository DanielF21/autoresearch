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
is counted and reported once at the end of the attempt, never raised.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

from autoresearch.config import ObserveConfig, RunConfig
from autoresearch.model.protocol import Message, ModelResponse
from autoresearch.types import AttemptRef, Prediction, StopReason

REQUIRED_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


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


# ----- Langfuse ------------------------------------------------------------------------


@dataclass
class LangfuseAttempt:
    """One trace. Turn n's input is only what was added to the conversation since
    turn n-1, because the first message holds the whole history and repeating it
    on all eighty turns would be megabytes of identical text."""

    _client: Any
    _root: Any
    _model: str
    _scope: Any = None
    _sent: int = 0
    _failures: list[str] = field(default_factory=list)

    def _try(self, what: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            self._failures.append(f"{what}: {e!r}")

    def box(self, box_id: str) -> None:
        self._try("box", lambda: self._root.update(metadata={"box_id": box_id}))

    def model_turn(self, turn: int, messages: list[Message], response: ModelResponse) -> None:
        def emit() -> None:
            new = messages[self._sent :]
            self._sent = len(messages)
            gen = self._root.start_observation(
                name=f"turn {turn}",
                as_type="generation",
                model=self._model,
                input=new,
            )
            gen.update(
                output={
                    "content": response.content,
                    "reasoning": response.reasoning,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in response.tool_calls
                    ],
                },
                usage_details={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                metadata={
                    "latency_s": round(response.latency_s, 3),
                    "finish_reason": response.finish_reason,
                    "cached_tokens": response.usage.cached_tokens,
                    "reasoning_tokens": response.usage.reasoning_tokens,
                },
            )
            gen.end()

        self._try(f"turn {turn}", emit)

    def tool(self, turn: int, name: str, args: dict[str, Any], result: str) -> None:
        # Unclipped, unlike the on disk transcript: the whole point of looking
        # here is to see what the agent actually got back.
        def emit() -> None:
            span = self._root.start_observation(name=name, as_type="tool", input=args)
            span.update(output=result, metadata={"turn": turn})
            span.end()

        self._try(f"tool {name}", emit)

    def end(
        self,
        stop: StopReason,
        *,
        patch: str | None = None,
        prediction: Prediction | None = None,
        error: str = "",
    ) -> None:
        output = {
            "stop_reason": str(stop),
            "has_patch": bool(patch),
            "patch_bytes": len(patch or ""),
            "predicted_speedup": None if prediction is None else prediction.speedup,
            "error": error,
        }
        self._try("end output", lambda: self._root.update(output=output))
        self._try("end trace io", lambda: self._root.set_trace_io(output=output))
        if self._failures:
            failures = list(self._failures)
            self._try(
                "end failures",
                lambda: self._root.update(metadata={"tracing_failures": failures}),
            )
        self._try("end span", self._root.end)
        # The attributes scope was entered when the attempt started and must be
        # left on the same thread that entered it, which is this one.
        if self._scope is not None:
            self._try("end scope", lambda: self._scope.__exit__(None, None, None))
        # The control box can be terminated as soon as a run ends, so nothing
        # may sit in the background queue waiting for a flush that never comes.
        self._try("flush", self._client.flush)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(self._failures)


@dataclass
class LangfuseTracer:
    """One per run. Each attempt roots its own trace, which is what we want:
    a round's workers are threads, and OpenTelemetry context is per thread."""

    _client: Any
    _model: str
    _session_id: str
    _attributes: Callable[..., Any] | None = None

    def attempt(self, ref: AttemptRef, run_id: str, base_sha: str) -> AttemptTrace:
        # Session and tags come from an attributes scope, and only spans created
        # inside it inherit them, so the scope is entered before the root span
        # and left in ``end``. Entering and leaving happen on the worker's own
        # thread, which is the whole of one attempt.
        scope = None
        if self._attributes is not None:
            try:
                scope = self._attributes(
                    session_id=self._session_id,
                    trace_name=f"{run_id} attempt {ref.dirname}",
                    tags=[run_id, f"round-{ref.round}"],
                )
                scope.__enter__()
            except Exception:
                scope = None
        try:
            root = self._client.start_observation(
                name=f"attempt {ref.dirname}",
                as_type="agent",
                input={"round": ref.round, "worker": ref.worker, "base_sha": base_sha},
            )
        except Exception:
            if scope is not None:
                with suppress(Exception):
                    scope.__exit__(None, None, None)
            return NullAttempt()
        return LangfuseAttempt(_client=self._client, _root=root, _model=self._model, _scope=scope)


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
        from langfuse import get_client, propagate_attributes

        client = get_client()
        if not client.auth_check():
            print("tracing off: langfuse rejected the keys", flush=True)
            return NullTracer()
    except Exception as e:
        print(f"tracing off: langfuse client failed: {e!r}", flush=True)
        return NullTracer()
    print(f"tracing on: session {session_id(observe, config.run_id)}", flush=True)
    return LangfuseTracer(
        _client=client,
        _model=config.worker.model,
        _session_id=session_id(observe, config.run_id),
        _attributes=propagate_attributes,
    )


def session_id(observe: ObserveConfig, run_id: str) -> str:
    return f"{observe.session_prefix}-{run_id}" if observe.session_prefix else run_id
