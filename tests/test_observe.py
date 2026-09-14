"""Tracing is a view, and a view must never be able to break the thing it views.

Two halves: the worker emits at the right moments on every exit path, and a
tracer that fails changes nothing about what the attempt produces.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autoresearch import observe
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, fail, ok
from autoresearch.config import ConfigError, ObserveConfig, RunConfig, load_config, parse_config
from autoresearch.model.fake_model import FakeChatModel, Scripted, text, tool_call
from autoresearch.model.protocol import ModelResponse, ToolCall
from autoresearch.types import AttemptRef, StopReason, Usage, WorkerOutput
from autoresearch.worker.agent_loop import AgentLoopWorker
from autoresearch.worker.protocol import WorkerInput
from tests.helpers import BASE_SHA, FakeTracer

ROOT = Path(__file__).parent.parent
DIFF = "diff --git a/x b/x\n--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n"


@pytest.fixture
def config() -> RunConfig:
    return load_config(ROOT / "configs" / "t1_w4d.toml")


def prepare(box: FakeBox, role: str) -> None:
    box.on("git checkout -q --detach", ok(f"{BASE_SHA}\npython 3.12.4\n"))
    box.on("git add -N", ok(DIFF))
    box.on("cat -n", ok("     1\tdef f(): pass\n"))


def make_input(config: RunConfig) -> WorkerInput:
    return WorkerInput(
        ref=AttemptRef(number=7, round=2, worker=1),
        base_sha=BASE_SHA,
        target=config.target,
        history=(),
        docs=(),
        cache_key="run-key",
    )


def run_attempt(
    config: RunConfig,
    script: list[Scripted],
    *,
    tracer: FakeTracer,
    box_prepare: Callable[[FakeBox, str], None] = prepare,
) -> WorkerOutput:
    factory = FakeBoxFactory(prepare=box_prepare)
    worker = AgentLoopWorker(FakeChatModel(script=script), factory, config, tracer)
    return worker.attempt(make_input(config))


def submit_script() -> list[Scripted]:
    """A function, not a constant: FakeChatModel pops its script as it goes."""
    return [
        tool_call("shell", {"cmd": "cat -n networkx/algorithms/cluster.py"}),
        tool_call("submit", {"predicted_speedup": 1.3, "rationale": "why"}),
    ]


def test_a_submitted_attempt_emits_one_trace_in_order(config: RunConfig) -> None:
    tracer = FakeTracer()
    run_attempt(config, submit_script(), tracer=tracer)

    trace = tracer.only
    assert trace.kinds == [
        "box",
        "model_turn",
        "tool",
        "model_turn",
        "tool",
        "end",
    ]
    assert trace.calls[-1] == ("end", "submitted")
    # Turn one sends the system and user messages; turn two sends only what was
    # added since, which is why a long history is not resent eighty times.
    assert [c[1][:2] for c in trace.calls if c[0] == "model_turn"] == [(1, 2), (2, 4)]
    assert [c[1] for c in trace.calls if c[0] == "tool"] == [(1, "shell"), (2, "submit")]


@pytest.mark.parametrize(
    ("script", "stop"),
    [
        ([text("no tools here"), text("still none")], "no_progress"),
        (
            [tool_call("shell", {"cmd": "cat -n a"})] * 10,
            "repeated_tool_call",
        ),
    ],
)
def test_every_ending_ends_the_trace(config: RunConfig, script: list[Scripted], stop: str) -> None:
    tracer = FakeTracer()
    run_attempt(config, script, tracer=tracer)
    assert tracer.only.kinds[-1] == "end"
    assert tracer.only.calls[-1] == ("end", stop)


def test_a_box_that_never_starts_still_ends_the_trace(config: RunConfig) -> None:
    """The check 2 failure was on an exit path, so every path is covered."""

    def broken(box: FakeBox, role: str) -> None:
        box.on("git checkout -q --detach", fail("no such commit"))

    tracer = FakeTracer()
    out = run_attempt(config, submit_script(), tracer=tracer, box_prepare=broken)
    assert out.stop_reason == StopReason.BOX_ERROR
    assert tracer.only.kinds == ["box", "end"]
    assert tracer.only.calls[-1] == ("end", "box_error")


def test_a_tracer_that_raises_changes_nothing(config: RunConfig) -> None:
    """A broken view must cost a trace, never the experiment."""
    good = run_attempt(config, submit_script(), tracer=FakeTracer())
    broken = run_attempt(config, submit_script(), tracer=FakeTracer(raises=True))
    assert broken.patch == good.patch
    assert broken.stop_reason == good.stop_reason
    assert broken.prediction == good.prediction


def test_the_default_worker_traces_nothing(config: RunConfig) -> None:
    factory = FakeBoxFactory(prepare=prepare)
    worker = AgentLoopWorker(FakeChatModel(script=submit_script()), factory, config)
    assert worker.attempt(make_input(config)).stop_reason == StopReason.SUBMITTED


# ----- config and construction ---------------------------------------------------------


def test_observe_section_defaults_to_off() -> None:
    text_without = (ROOT / "configs" / "t1_w4d.toml").read_text().split("[observe]")[0]
    assert parse_config(text_without).observe == ObserveConfig(enabled=False)


def test_observe_section_is_validated() -> None:
    base = (ROOT / "configs" / "t1_w4d.toml").read_text().split("[observe]")[0]
    with pytest.raises(ConfigError, match="enabled"):
        parse_config(base + '[observe]\nenabled = "yes"\n')
    with pytest.raises(ConfigError, match="session_prefix"):
        parse_config(base + "[observe]\nenabled = true\nsession_prefix = 3\n")


def test_tracing_off_without_keys(config: RunConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing keys must not stop a run from starting, only from being watched."""
    for key in observe.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert isinstance(observe.build_tracer(config), observe.NullTracer)


def test_tracing_on_with_the_sail_key(config: RunConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAIL_API_KEY", "sk_test")
    tracer = observe.build_tracer(config)
    assert isinstance(tracer, observe.VoyageTracer)
    assert tracer._series == config.run_id


def test_session_id_groups_a_setting() -> None:
    assert observe.session_id(ObserveConfig(True), "t1_w1") == "t1_w1"
    assert observe.session_id(ObserveConfig(True, "pilot"), "t1_w1") == "pilot-t1_w1"


def test_null_tracer_accepts_every_call() -> None:
    trace = observe.NullTracer().attempt(AttemptRef(1, 1, 0), "t1_w1", BASE_SHA)
    trace.box("sb_1")
    trace.tool(1, "shell", {"cmd": "cat a"}, "contents")
    trace.end(StopReason.SUBMITTED, patch=DIFF)


# ----- Sail Voyages, against a stand in for the SDK so no network is used ---------------


REF = AttemptRef(number=7, round=2, worker=1)


def response() -> ModelResponse:
    return ModelResponse(
        content="looking",
        reasoning="thinking",
        tool_calls=(ToolCall("c1", "shell", {"cmd": "cat a"}),),
        usage=Usage(1000, 600, 100, 40),
        finish_reason="tool_calls",
        latency_s=1.23456,
    )


def start(sdk: "_FakeSdk") -> observe.VoyageAttempt:
    tracer = observe.VoyageTracer(_sdk=sdk, _series="t1_w1", _model="deepseek/v4")
    trace = tracer.attempt(REF, "t1_w1", BASE_SHA)
    assert isinstance(trace, observe.VoyageAttempt)
    return trace


def test_the_voyage_attempt_records_events_and_completes() -> None:
    sdk = _FakeSdk()
    trace = start(sdk)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    trace.box("sb_1")
    trace.model_turn(1, messages, response())
    trace.tool(1, "shell", {"cmd": "cat a"}, "x" * 10_000)
    trace.end(StopReason.SUBMITTED, patch=DIFF)

    voyage = sdk.only
    assert voyage.name == "t1_w1"
    assert voyage.metadata["attempt"] == REF.dirname and voyage.metadata["model"] == "deepseek/v4"
    assert [k for k, _ in voyage.events] == ["box.ready", "model.turn", "tool.result"]
    turn = voyage.events[1][1]
    assert turn["new_messages"] == 2 and turn["latency_s"] == 1.235
    assert turn["input_tokens"] == response().usage.prompt_tokens
    assert turn["tool_calls"] == [{"name": "shell", "arguments": '{"cmd": "cat a"}'}]
    tool = voyage.events[2][1]
    assert tool["result_chars"] == 10_000 and len(tool["result"]) < 4100
    assert voyage.agent_log == [("enter", "Worker", "worker"), ("exit",)]
    assert voyage.terminal == [("complete", "submitted", voyage.terminal[0][2])]
    assert voyage.terminal[0][2]["stop_reason"] == "submitted"
    assert voyage.terminal[0][2]["patch_bytes"] == len(DIFF)
    assert trace.failures == ()


def test_the_fallback_voyage_is_disabled_as_soon_as_it_is_created() -> None:
    """Otherwise every thread without its own Voyage, the referee's included,
    would attach its box commands and model calls to this attempt. The worker's
    own thread is disabled first, so its model calls never resolve the fallback
    while another attempt is between its create and its disable."""
    sdk = _FakeSdk()
    start(sdk)
    assert sdk.calls == ["disable", "create", "disable"]
    assert sdk.disabled_on[0] == threading.get_ident()
    assert sdk.disabled_on[1] != threading.get_ident()


def test_a_create_that_finishes_after_the_bound_gave_up_is_failed_not_left_running() -> None:
    released = threading.Event()
    sdk = _FakeSdk(create_hook=lambda: released.wait(30))
    tracer = observe.VoyageTracer(_sdk=sdk, _series="t1_w1", _model="m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(observe, "FLUSH_TIMEOUT_S", 0.2)
        trace = tracer.attempt(REF, "t1_w1", BASE_SHA)
    assert isinstance(trace, observe.NullAttempt)
    released.set()
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline and not (sdk.voyages and sdk.voyages[0].terminal):
        time.sleep(0.01)
    assert sdk.only.terminal[0][:2] == ("fail", "tracing_abandoned")
    assert sdk.calls[-1] == "disable"


def test_the_sdk_auto_spans_are_switched_off(
    config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SAIL_API_KEY", "sk_test")
    monkeypatch.delenv("SAIL_VOYAGE_AUTO_SPANS", raising=False)
    observe.build_tracer(config)
    import os

    assert os.environ["SAIL_VOYAGE_AUTO_SPANS"] == "0"


def test_turns_count_only_the_messages_added_since_the_last() -> None:
    sdk = _FakeSdk()
    trace = start(sdk)
    trace.model_turn(1, [{}, {}], response())
    trace.model_turn(2, [{}, {}, {}, {}], response())
    assert [p["new_messages"] for _, p in sdk.only.events] == [2, 2]


def test_an_attempt_that_ends_in_error_fails_the_voyage() -> None:
    sdk = _FakeSdk()
    trace = start(sdk)
    trace.end(StopReason.MODEL_ERROR, error="upstream 503")
    kind, error_type, message, payload = sdk.only.terminal[0]
    assert (kind, error_type, message) == ("fail", "model_error", "upstream 503")
    assert payload["error"] == "upstream 503"


def test_a_create_that_fails_traces_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    sdk = _FakeSdk(create_raises=True)
    tracer = observe.VoyageTracer(_sdk=sdk, _series="t1_w1", _model="m")
    assert isinstance(tracer.attempt(REF, "t1_w1", BASE_SHA), observe.NullAttempt)
    assert "tracing: start" in capsys.readouterr().out


def test_a_create_that_hangs_does_not_hold_up_the_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observe, "FLUSH_TIMEOUT_S", 0.2)
    released = threading.Event()
    sdk = _FakeSdk(create_hook=lambda: released.wait(30))
    tracer = observe.VoyageTracer(_sdk=sdk, _series="t1_w1", _model="m")

    t0 = time.perf_counter()
    trace = tracer.attempt(REF, "t1_w1", BASE_SHA)
    waited = time.perf_counter() - t0
    released.set()  # let the daemon thread go before the test ends

    assert waited < 5.0, f"attempt() blocked for {waited:.1f}s on a hanging create"
    assert isinstance(trace, observe.NullAttempt)


def test_a_terminal_flush_that_hangs_does_not_hold_up_the_attempt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The t1_w4b regression: a slow flush raises nothing, so try/except missed it.

    Three of twelve workers lost about forty seconds each to the exporter
    retrying a read timeout. Observability may cost a trace, never the run.
    """
    monkeypatch.setattr(observe, "FLUSH_TIMEOUT_S", 0.2)
    released = threading.Event()
    sdk = _FakeSdk(terminal_hook=lambda: released.wait(30))
    trace = start(sdk)

    t0 = time.perf_counter()
    trace.end(StopReason.SUBMITTED, patch=DIFF)
    waited = time.perf_counter() - t0
    released.set()

    assert waited < 5.0, f"end() blocked for {waited:.1f}s on a hanging flush"
    assert any("abandoned" in f for f in trace.failures), trace.failures
    assert "tracing: end" in capsys.readouterr().out
    assert sdk.only.agent_log[-1] == ("exit",)  # the agent is still left; only the send is not


def test_a_failing_sdk_is_recorded_and_swallowed() -> None:
    sdk = _FakeSdk(events_raise=True)
    trace = start(sdk)
    trace.tool(1, "shell", {"cmd": "cat a"}, "contents")
    trace.end(StopReason.SUBMITTED)
    assert any("tool.result" in f for f in trace.failures)
    assert sdk.only.terminal[0][2]["tracing_failures"] == list(trace.failures)


class _FakeAgent:
    def __init__(self, voyage: "_FakeVoyage", name: str, role: str | None) -> None:
        self.voyage = voyage
        self.name = name
        self.role = role

    def __enter__(self) -> "_FakeAgent":
        self.voyage.agent_log.append(("enter", self.name, self.role))
        return self

    def __exit__(self, *exc: object) -> None:
        self.voyage.agent_log.append(("exit",))


class _FakeVoyage:
    def __init__(self, sdk: "_FakeSdk", name: str, metadata: dict[str, Any]) -> None:
        self.sdk = sdk
        self.id = "voy_1"
        self.dashboard_url = "https://app.sailresearch.com/prod/voyages/voy_1"
        self.name = name
        self.metadata = metadata
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.agent_log: list[tuple[object, ...]] = []
        self.terminal: list[tuple[Any, ...]] = []

    def event(self, kind: str, *, payload: dict[str, Any]) -> None:
        if self.sdk.events_raise:
            raise RuntimeError("voyages are down")
        self.events.append((kind, payload))

    def agent(self, name: str, *, role: str | None = None) -> _FakeAgent:
        return _FakeAgent(self, name, role)

    def complete(self, message: str | None = None, payload: dict[str, Any] | None = None) -> None:
        self.sdk.terminal_hook()
        self.terminal.append(("complete", message, payload))

    def fail(
        self,
        error_type: str = "harness_error",
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.sdk.terminal_hook()
        self.terminal.append(("fail", error_type, message, payload))


class _FakeSdk:
    """Stands in for the ``sail.voyage`` module."""

    def __init__(
        self,
        *,
        create_raises: bool = False,
        events_raise: bool = False,
        create_hook: Callable[[], object] = lambda: None,
        terminal_hook: Callable[[], object] = lambda: None,
    ) -> None:
        self.create_raises = create_raises
        self.events_raise = events_raise
        self.create_hook = create_hook
        self.terminal_hook = terminal_hook
        self.calls: list[str] = []
        self.disabled_on: list[int] = []
        self.voyages: list[_FakeVoyage] = []

    def create(self, name: str, *, metadata: dict[str, Any]) -> _FakeVoyage:
        self.calls.append("create")
        self.create_hook()
        if self.create_raises:
            raise RuntimeError("voyages are down")
        voyage = _FakeVoyage(self, name, metadata)
        self.voyages.append(voyage)
        return voyage

    def disable(self) -> None:
        self.calls.append("disable")
        self.disabled_on.append(threading.get_ident())

    @property
    def only(self) -> _FakeVoyage:
        assert len(self.voyages) == 1, f"expected one voyage, got {len(self.voyages)}"
        return self.voyages[0]
