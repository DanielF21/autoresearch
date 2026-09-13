"""A minimal agent loop for the Scribe's roles.

The same rules as the worker's loop in ``worker/agent_loop.py``: turn, wall clock
and input token caps, a stop on the same call three times running, one nudge and
then a stop on a second turn with no tool call, and a model error ends the run.
It is a copy rather than a shared function because the worker loop is tied to
boxes; merging the two is a later refactor, not a behaviour change.

A role ends by calling its submit tool with arguments that pass validation. What
it submitted is returned as data, and the transcript records every turn.
"""

from __future__ import annotations

import enum
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from autoresearch.model.protocol import ChatModel, Message, ModelError, ToolCall
from autoresearch.scribe.config import RoleModel
from autoresearch.scribe.tools import Tool, ToolContext, execute
from autoresearch.types import Usage

REPEAT_LIMIT = 3
NO_PROGRESS_LIMIT = 2
TRANSCRIPT_CLIP = 4000
NUDGE = (
    "A reply without a tool call is not read. Call a tool, or call the submit tool "
    "with your answer."
)


class LoopStop(enum.StrEnum):
    SUBMITTED = "submitted"
    MAX_TURNS = "max_turns"
    MAX_SECONDS = "max_seconds"
    MAX_INPUT_TOKENS = "max_input_tokens"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    NO_PROGRESS = "no_progress"
    MODEL_ERROR = "model_error"


@dataclass(frozen=True)
class Caps:
    max_turns: int
    max_seconds: int
    max_input_tokens: int

    @classmethod
    def of(cls, role: RoleModel) -> Caps:
        return cls(role.max_turns, role.max_seconds, role.max_input_tokens)


@dataclass
class Transcript:
    lines: list[str] = field(default_factory=list)

    def add(self, **record: Any) -> None:
        record["t"] = time.time()
        self.lines.append(json.dumps(record, sort_keys=True))

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")


@dataclass(frozen=True)
class LoopResult:
    stop: LoopStop
    submitted: dict[str, Any] | None
    usage: Usage
    turns: int
    transcript: str
    error: str
    wall_s: float


def _clip(s: str, n: int = TRANSCRIPT_CLIP) -> str:
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more]"


def _signature(call: ToolCall) -> str:
    return call.name + ":" + json.dumps(call.arguments, sort_keys=True)


def run_agent(
    model: ChatModel,
    messages: list[Message],
    tools: Sequence[Tool],
    ctx: ToolContext,
    caps: Caps,
    *,
    cache_key: str,
    clock: Callable[[], float] = time.perf_counter,
) -> LoopResult:
    """Run until a submit tool accepts, or a cap trips. ``messages`` is extended in place."""
    t0 = clock()
    transcript = Transcript()
    usage = Usage()
    turns = 0
    recent: list[str] = []
    idle = 0
    specs = [t.spec for t in tools]

    def finish(
        stop: LoopStop, submitted: dict[str, Any] | None = None, error: str = ""
    ) -> LoopResult:
        transcript.add(kind="end", stop=str(stop), turns=turns, error=error)
        return LoopResult(stop, submitted, usage, turns, transcript.text(), error, clock() - t0)

    transcript.add(kind="start", messages=[_clip(str(m.get("content", ""))) for m in messages])
    while True:
        if turns >= caps.max_turns:
            return finish(LoopStop.MAX_TURNS)
        if clock() - t0 > caps.max_seconds:
            return finish(LoopStop.MAX_SECONDS)
        if usage.prompt_tokens > caps.max_input_tokens:
            return finish(LoopStop.MAX_INPUT_TOKENS)

        turns += 1
        try:
            resp = model.complete(messages, specs, cache_key=cache_key)
        except ModelError as e:
            return finish(LoopStop.MODEL_ERROR, error=str(e))
        usage = usage + resp.usage
        messages.append(resp.message)
        transcript.add(
            kind="model",
            turn=turns,
            latency_s=round(resp.latency_s, 3),
            usage=resp.usage.to_dict(),
            finish=resp.finish_reason,
            reasoning=_clip(resp.reasoning),
            content=_clip(resp.content),
            tool_calls=[{"name": c.name, "args": c.arguments} for c in resp.tool_calls],
        )

        if not resp.tool_calls:
            idle += 1
            if idle >= NO_PROGRESS_LIMIT:
                return finish(LoopStop.NO_PROGRESS)
            messages.append({"role": "user", "content": NUDGE})
            continue
        idle = 0

        for call in resp.tool_calls:
            recent = [*recent, _signature(call)][-REPEAT_LIMIT:]
            if len(recent) == REPEAT_LIMIT and len(set(recent)) == 1:
                return finish(LoopStop.REPEATED_TOOL_CALL)
            result = execute(tools, ctx, call.name, call.arguments)
            transcript.add(kind="tool", turn=turns, name=call.name, result=_clip(result.text))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})
            if result.submitted is not None:
                return finish(LoopStop.SUBMITTED, submitted=result.submitted)
