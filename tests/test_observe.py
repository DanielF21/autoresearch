"""Tracing is a view, and a view must never be able to break the thing it views.

Two halves: the worker emits at the right moments on every exit path, and a
tracer that fails changes nothing about what the attempt produces.
"""

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from autoresearch import observe
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, fail, ok
from autoresearch.config import ConfigError, ObserveConfig, RunConfig, load_config, parse_config
from autoresearch.model.fake_model import FakeChatModel, Scripted, text, tool_call
from autoresearch.types import AttemptRef, StopReason, WorkerOutput
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


def test_session_id_groups_a_setting() -> None:
    assert observe.session_id(ObserveConfig(True), "t1_w1") == "t1_w1"
    assert observe.session_id(ObserveConfig(True, "pilot"), "t1_w1") == "pilot-t1_w1"


def test_null_tracer_accepts_every_call() -> None:
    trace = observe.NullTracer().attempt(AttemptRef(1, 1, 0), "t1_w1", BASE_SHA)
    trace.box("sb_1")
    trace.tool(1, "shell", {"cmd": "cat a"}, "contents")
    trace.end(StopReason.SUBMITTED, patch=DIFF)


def test_the_langfuse_attempt_maps_turns_and_tools_to_observations() -> None:
    """Against a stand in for the SDK, so the shape is pinned without a network."""
    client = _FakeClient()
    root = client.start_observation(name="attempt 0007", as_type="span", input={})
    trace = observe.LangfuseAttempt(_client=client, _root=root, _model="deepseek/v4")

    trace.box("sb_1")
    trace.tool(1, "shell", {"cmd": "cat a"}, "contents")
    trace.end(StopReason.SUBMITTED, patch=DIFF)

    assert [(c.name, c.as_type) for c in root.children] == [("shell", "tool")]
    assert root.children[0].output == "contents"
    assert root.ended and client.flushed == 1
    assert json.loads(json.dumps(root.output))["stop_reason"] == "submitted"
    assert trace.failures == ()


def test_a_flush_that_hangs_does_not_hold_up_the_attempt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The t1_w4b regression: a slow flush raises nothing, so try/except missed it.

    Three of twelve workers lost about forty seconds each to the exporter
    retrying a read timeout. Observability may cost a trace, never the run.
    """
    monkeypatch.setattr(observe, "FLUSH_TIMEOUT_S", 0.2)
    released = threading.Event()
    client = _FakeClient()

    def hang() -> None:
        released.wait(30)

    client.flush = hang  # type: ignore[method-assign]
    root = client.start_observation(name="attempt 0009", as_type="span", input={})
    trace = observe.LangfuseAttempt(_client=client, _root=root, _model="m")

    t0 = time.perf_counter()
    trace.end(StopReason.SUBMITTED, patch=DIFF)
    waited = time.perf_counter() - t0
    released.set()  # let the daemon thread go before the test ends

    assert waited < 5.0, f"end() blocked for {waited:.1f}s on a hanging flush"
    assert any("abandoned" in f for f in trace.failures), trace.failures
    assert "tracing: flush" in capsys.readouterr().out
    assert root.ended  # the span is still closed; only the send was given up on


def test_a_flush_that_returns_is_not_reported_as_abandoned() -> None:
    client = _FakeClient()
    root = client.start_observation(name="attempt 0001", as_type="span", input={})
    trace = observe.LangfuseAttempt(_client=client, _root=root, _model="m")
    trace.end(StopReason.SUBMITTED)
    assert client.flushed == 1 and trace.failures == ()


def test_a_failing_sdk_is_recorded_and_swallowed() -> None:
    client = _FakeClient()
    root = _Observation("attempt", "span", broken=True)
    trace = observe.LangfuseAttempt(_client=client, _root=root, _model="m")
    trace.tool(1, "shell", {"cmd": "cat a"}, "contents")
    trace.end(StopReason.SUBMITTED)
    assert any("shell" in f for f in trace.failures)


class _Observation:
    def __init__(self, name: str, as_type: str, broken: bool = False) -> None:
        self.name = name
        self.as_type = as_type
        self.broken = broken
        self.children: list[_Observation] = []
        self.output: object = None
        self.metadata: dict[str, object] = {}
        self.trace_io: list[dict[str, object]] = []
        self.ended = False

    def _check(self) -> None:
        if self.broken:
            raise RuntimeError("langfuse is down")

    def start_observation(self, *, name: str, as_type: str, **kw: object) -> "_Observation":
        self._check()
        child = _Observation(name, as_type)
        self.children.append(child)
        return child

    def update(self, **kw: object) -> None:
        self._check()
        if "output" in kw:
            self.output = kw["output"]
        extra = kw.get("metadata")
        if isinstance(extra, dict):
            self.metadata.update(extra)

    def set_trace_io(self, **kw: object) -> None:
        self._check()
        self.trace_io.append(dict(kw))

    def end(self) -> None:
        self._check()
        self.ended = True


class _FakeClient:
    def __init__(self) -> None:
        self.roots: list[_Observation] = []
        self.flushed = 0

    def start_observation(self, *, name: str, as_type: str, **kw: object) -> _Observation:
        root = _Observation(name, as_type)
        self.roots.append(root)
        return root

    def flush(self) -> None:
        self.flushed += 1
