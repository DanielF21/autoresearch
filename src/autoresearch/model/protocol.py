"""The narrow view of a chat model the worker needs: messages and tools in, one
response out. Messages use the OpenAI chat shape as plain dicts because that is
what Sail's endpoint takes and returns, and the assistant message is echoed back
verbatim on the next turn, reasoning content included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from autoresearch.types import Usage

Message = dict[str, Any]
ToolSpec = dict[str, Any]


class ModelError(RuntimeError):
    """The request failed after any retry. The attempt ends with MODEL_ERROR."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(frozen=True)
class ModelResponse:
    """One assistant turn. ``message`` is the dict to append to the conversation."""

    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    finish_reason: str
    latency_s: float
    message: Message = field(default_factory=dict)


class ChatModel(Protocol):
    def complete(
        self, messages: list[Message], tools: list[ToolSpec], *, cache_key: str
    ) -> ModelResponse: ...
