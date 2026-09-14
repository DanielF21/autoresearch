"""A scripted chat model for tests. Returns its responses in order and records
every request so a test can assert what the worker sent."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autoresearch.model.protocol import Message, ModelError, ModelResponse, ToolCall, ToolSpec
from autoresearch.types import Usage

Scripted = ModelResponse | Callable[[list[Message]], ModelResponse] | Exception


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "") -> ModelResponse:
    """A response that asks for one tool."""
    cid = call_id or f"call_{name}_{abs(hash(json.dumps(arguments, sort_keys=True))) % 10000}"
    raw = json.dumps(arguments)
    return ModelResponse(
        content="",
        reasoning="thinking",
        tool_calls=(ToolCall(id=cid, name=name, arguments=arguments, raw_arguments=raw),),
        usage=Usage(prompt_tokens=100, cached_tokens=50, completion_tokens=20, reasoning_tokens=5),
        finish_reason="tool_calls",
        latency_s=0.01,
        message={
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking",
            "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": raw}}
            ],
        },
    )


def malformed_call(name: str, raw: str, call_id: str = "call_cut") -> ModelResponse:
    """A response whose tool call arguments are not JSON, as ``parse_response`` returns it:
    empty arguments, the raw text kept, and the echo repaired to ``{}``."""
    return ModelResponse(
        content="",
        reasoning="thinking",
        tool_calls=(
            ToolCall(id=call_id, name=name, arguments={}, raw_arguments=raw, malformed=True),
        ),
        usage=Usage(prompt_tokens=100, cached_tokens=50, completion_tokens=20, reasoning_tokens=5),
        finish_reason="tool_calls",
        latency_s=0.01,
        message={
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
            ],
        },
    )


def text(content: str) -> ModelResponse:
    """A response with no tool call."""
    return ModelResponse(
        content=content,
        reasoning="",
        tool_calls=(),
        usage=Usage(prompt_tokens=100, cached_tokens=0, completion_tokens=10, reasoning_tokens=0),
        finish_reason="stop",
        latency_s=0.01,
        message={"role": "assistant", "content": content},
    )


@dataclass
class FakeChatModel:
    script: list[Scripted] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)
    cache_keys: list[str] = field(default_factory=list)
    offered: list[list[str]] = field(default_factory=list)  # tool names, per request

    def complete(
        self, messages: list[Message], tools: list[ToolSpec], *, cache_key: str
    ) -> ModelResponse:
        self.requests.append([dict(m) for m in messages])
        self.cache_keys.append(cache_key)
        self.offered.append([t["function"]["name"] for t in tools])
        if not self.script:
            raise ModelError("fake model script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        return item
