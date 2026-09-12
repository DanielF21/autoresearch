"""The only module that talks to Sail inference.

One request per turn, with the settings verified on 2026-09-11: tool calling
through ``tools``, thinking through ``reasoning_effort``, the direct tier
through ``metadata.completion_window``, and prefix caching through
``prompt_cache_key``. Sail makes exactly one attempt per request, so the single
retry here carries an idempotency key and only fires on failures the server
labels as transient.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from autoresearch.config import WorkerConfig
from autoresearch.model.protocol import Message, ModelError, ModelResponse, ToolCall, ToolSpec
from autoresearch.types import Usage

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}


def parse_response(raw: dict[str, Any], latency_s: float) -> ModelResponse:
    """Turn the endpoint's JSON into a ModelResponse. Pure, so it is tested."""
    try:
        choice = raw["choices"][0]
        message = dict(choice["message"])
    except (KeyError, IndexError, TypeError) as e:
        raise ModelError(f"response has no message: {json.dumps(raw)[:500]}") from e

    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                args = {"value": args}
        except json.JSONDecodeError:
            args = {}
        calls.append(
            ToolCall(
                id=str(tc.get("id", "")),
                name=str(fn.get("name", "")),
                arguments=args,
                raw_arguments=raw_args,
            )
        )

    u = raw.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(u.get("prompt_tokens", 0)),
        cached_tokens=int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
        completion_tokens=int(u.get("completion_tokens", 0)),
        reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)),
    )
    # The assistant message goes back into the conversation as received, so the
    # model sees its own reasoning and tool calls on the next turn.
    echo: Message = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("reasoning_content"):
        echo["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        echo["tool_calls"] = message["tool_calls"]
    return ModelResponse(
        content=str(message.get("content") or ""),
        reasoning=str(message.get("reasoning_content") or ""),
        tool_calls=tuple(calls),
        usage=usage,
        finish_reason=str(choice.get("finish_reason", "")),
        latency_s=latency_s,
        message=echo,
    )


class SailChatModel:
    def __init__(self, config: WorkerConfig) -> None:
        import sail

        self._sail = sail
        self._config = config

    def _payload(
        self, messages: list[Message], tools: list[ToolSpec], cache_key: str
    ) -> dict[str, Any]:
        c = self._config
        payload: dict[str, Any] = {
            "model": c.model,
            "messages": messages,
            "prompt_cache_key": cache_key,
            "metadata": {"completion_window": c.completion_window},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if c.reasoning_effort and c.reasoning_effort != "none":
            payload["reasoning_effort"] = c.reasoning_effort
        return payload

    def complete(
        self, messages: list[Message], tools: list[ToolSpec], *, cache_key: str
    ) -> ModelResponse:
        payload = self._payload(messages, tools, cache_key)
        key = f"autoresearch-{uuid.uuid4().hex}"
        last: Exception | None = None
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                raw = self._sail.inference.chat.completions.create(
                    headers={"Idempotency-Key": key},
                    timeout=float(self._config.turn_timeout),
                    **payload,
                )
                return parse_response(raw, time.perf_counter() - t0)
            except Exception as e:
                last = e
                if attempt == 0 and _is_transient(e):
                    time.sleep(2.0)
                    continue
                break
        raise ModelError(f"inference failed: {last!r}") from last


def _is_transient(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS
    return bool(getattr(e, "retryable", False))
