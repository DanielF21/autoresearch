"""The pure parts of the Sail model client: payload shape and response parsing.
The network path is exercised by Phase 2b."""

from pathlib import Path

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


def test_parse_response_without_choices_raises() -> None:
    with pytest.raises(ModelError):
        parse_response({"error": "nope"}, 0.0)


def test_payload_carries_the_verified_settings() -> None:
    cfg = load_config(ROOT / "configs" / "t1_w1.toml").worker
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
