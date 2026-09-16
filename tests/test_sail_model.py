"""The pure parts of the Sail model client: payload shape and response parsing.
The network path is exercised by Phase 2b."""

from pathlib import Path
from typing import Any

import pytest

from autoresearch.config import load_config
from autoresearch.model.protocol import ModelError
from autoresearch.model.sail_model import SailChatModel, _is_transient, parse_response
from autoresearch.types import Usage

ROOT = Path(__file__).parent.parent

RAW = {
    "choices": [
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "let me look",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"cmd": "cat a.py"}'},
                    }
                ],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 387,
        "prompt_tokens_details": {"cached_tokens": 256},
        "completion_tokens": 126,
        "completion_tokens_details": {"reasoning_tokens": 76},
    },
}


def test_parse_response_reads_tool_calls_usage_and_echo() -> None:
    r = parse_response(RAW, 1.5)
    assert r.tool_calls[0].name == "shell"
    assert r.tool_calls[0].arguments == {"cmd": "cat a.py"}
    assert r.tool_calls[0].id == "call_1"
    assert r.usage == Usage(387, 256, 126, 76)
    assert r.reasoning == "let me look"
    assert r.finish_reason == "tool_calls"
    assert r.latency_s == 1.5
    assert r.message["role"] == "assistant"
    assert r.message["reasoning_content"] == "let me look"
    assert r.message["tool_calls"] == RAW["choices"][0]["message"]["tool_calls"]  # type: ignore[index]


def test_parse_response_plain_text_has_no_reasoning_key() -> None:
    raw = {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {},
    }
    r = parse_response(raw, 0.1)
    assert r.content == "hi" and r.tool_calls == () and r.usage == Usage()
    assert "reasoning_content" not in r.message and "tool_calls" not in r.message


def test_parse_response_bad_json_arguments_become_empty_dict() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "c", "function": {"name": "shell", "arguments": "{oops"}}]
                }
            }
        ]
    }
    r = parse_response(raw, 0.0)
    assert r.tool_calls[0].arguments == {} and r.tool_calls[0].raw_arguments == "{oops"
    assert r.tool_calls[0].malformed
    # The echo carries "{}" in place of the cut off text: the endpoint refuses a
    # request whose history holds a tool call with invalid JSON arguments, which
    # would end the conversation at every later turn.
    echoed = r.message["tool_calls"][0]["function"]
    assert echoed == {"name": "shell", "arguments": "{}"}
    assert (
        "{oops" in r.tool_calls[0].malformed_reply
        and "did not run" in r.tool_calls[0].malformed_reply
    )
    good = parse_response(RAW, 0.0)
    assert not good.tool_calls[0].malformed


def test_parse_response_without_choices_raises() -> None:
    with pytest.raises(ModelError):
        parse_response({"error": "nope"}, 0.0)


def test_payload_carries_the_verified_settings() -> None:
    cfg = load_config(ROOT / "configs" / "t1_w4d.toml").worker
    model = SailChatModel.__new__(SailChatModel)
    model._config = cfg
    payload = model._payload([{"role": "user", "content": "x"}], [{"type": "function"}], "key")
    assert payload["model"] == "deepseek/deepseek-v4-pro-0813"
    assert payload["reasoning_effort"] == "high"
    assert payload["metadata"] == {"completion_window": "asap"}
    assert payload["prompt_cache_key"] == "key"
    assert payload["tool_choice"] == "auto"


class _FakeError(Exception):
    def __init__(self, status: int | None = None, retryable: bool = False) -> None:
        self.status_code = status
        self.retryable = retryable


def test_transient_classification() -> None:
    assert _is_transient(_FakeError(503))
    assert _is_transient(_FakeError(529))
    assert not _is_transient(_FakeError(400))
    assert _is_transient(_FakeError(None, retryable=True))
    assert not _is_transient(RuntimeError("x"))


class _Completions:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls: list[str] = []

    def create(self, **kwargs: object) -> Any:
        headers = kwargs["headers"]
        assert isinstance(headers, dict)
        self.calls.append(str(headers["Idempotency-Key"]))
        if len(self.calls) <= self.failures:
            raise _FakeError(None, retryable=True)
        return RAW


def _client(
    retries: int, failures: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[SailChatModel, _Completions]:
    from dataclasses import replace
    from types import SimpleNamespace

    monkeypatch.setattr("autoresearch.model.sail_model.time.sleep", lambda _s: None)
    completions = _Completions(failures)
    model = SailChatModel.__new__(SailChatModel)
    cfg = load_config(ROOT / "configs" / "t1_w4d.toml").worker
    model._config = replace(cfg, inference_retries=retries)
    model._sail = SimpleNamespace(
        inference=SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    )
    return model, completions


def test_a_transient_failure_is_retried_up_to_the_configured_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, completions = _client(retries=3, failures=3, monkeypatch=monkeypatch)
    assert model.complete([], [], cache_key="k").usage.prompt_tokens == 387
    assert len(completions.calls) == 4 and len(set(completions.calls)) == 1  # one idempotency key

    model, completions = _client(retries=3, failures=4, monkeypatch=monkeypatch)
    with pytest.raises(ModelError):
        model.complete([], [], cache_key="k")
    assert len(completions.calls) == 4

    model, completions = _client(retries=1, failures=1, monkeypatch=monkeypatch)
    model.complete([], [], cache_key="k")
    assert len(completions.calls) == 2  # the harness default: one retry
