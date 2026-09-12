"""The agent loop against a scripted model and a fake box factory."""

import json
from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, fail, ok
from autoresearch.boxes.image import REPO_DIR
from autoresearch.config import RunConfig, load_config
from autoresearch.model.fake_model import FakeChatModel, Scripted, text, tool_call
from autoresearch.model.protocol import ModelError
from autoresearch.types import Attempt as HistoryAttempt
from autoresearch.types import (
    AttemptRef,
    Measurement,
    Prediction,
    StopReason,
    SuiteResult,
    Usage,
)
from autoresearch.worker.agent_loop import AgentLoopWorker
from autoresearch.worker.protocol import WorkerInput
from autoresearch.worker.tools import BASE_DIR, HISTORY_DIR
from tests.helpers import BASE_SHA

ROOT = Path(__file__).parent.parent
DIFF = "diff --git a/networkx/algorithms/cluster.py b/networkx/algorithms/cluster.py\n--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n"


@pytest.fixture
def config() -> RunConfig:
    return load_config(ROOT / "configs" / "t1_w1.toml")


def prepare(box: FakeBox, role: str) -> None:
    """A worker box whose setup succeeds and whose git diff returns DIFF."""
    assert role == "worker"
    box.on("git checkout -q --detach", ok(f"{BASE_SHA}\n"))
    box.on("git add -N", ok(DIFF))
    box.on("awk", ok("     1\tdef f(): pass\n"))
    box.on(
        "run_tests.py",
        ok(json.dumps({"ok": True, "passed": 3, "failed": 0, "errors": 0, "duration_s": 1})),
    )


def make_input(config: RunConfig, history: tuple[HistoryAttempt, ...] = ()) -> WorkerInput:
    return WorkerInput(
        ref=AttemptRef(number=len(history) + 1, round=1, worker=0),
        base_sha=BASE_SHA,
        target=config.target,
        history=history,
        docs=(("profile.txt", "cluster.py:160 82% self time"),),
        cache_key="run-key",
    )


def submit_script() -> list[Scripted]:
    return [
        tool_call("read_file", {"path": "networkx/algorithms/cluster.py"}),
        tool_call("edit_file", {"path": "networkx/algorithms/cluster.py", "old": "a", "new": "b"}),
        tool_call("run_tests", {"scope": "module"}),
        tool_call("submit", {"predicted_speedup": 1.3, "rationale": "precompute neighbour sets"}),
    ]


def test_happy_path_submits_a_diff(config: RunConfig) -> None:
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(script=submit_script())
    worker = AgentLoopWorker(model, factory, config)
    box_holder: list[FakeBox] = []

    def prep_and_edit(box: FakeBox, role: str) -> None:
        prepare(box, role)
        box.write(f"{REPO_DIR}/networkx/algorithms/cluster.py", b"a\n")
        box_holder.append(box)

    factory.prepare = prep_and_edit
    out = worker.attempt(make_input(config))

    assert out.stop_reason == StopReason.SUBMITTED
    assert out.patch == DIFF
    assert out.prediction == Prediction(1.3)
    assert out.rationale == "precompute neighbour sets"
    assert out.turns == 4
    assert out.usage == Usage(400, 200, 80, 20)
    assert out.box_id == "sb_fake_1"
    box = box_holder[0]
    assert box.terminated
    assert box.read(f"{REPO_DIR}/networkx/algorithms/cluster.py") == b"b\n"
    assert "/workspace/incumbent.diff" not in box.files
    setup_cmd = next(c for c in box.commands if "git checkout -q --detach" in c)
    assert f"--detach {BASE_SHA} " in setup_cmd
    assert "git apply" not in setup_cmd
    assert any(BASE_DIR in c for c in box.commands)
    assert ("/workspace/guest/provenance.py") in box.files

    # A git pathspec is repository relative. Naming the sibling baseline
    # worktree by absolute path made git refuse the diff and lost a whole
    # attempt in check 2, so the collect command carries no pathspec at all.
    collect = next(c for c in box.commands if "git diff" in c)
    assert collect == f"cd {REPO_DIR} && git add -N . && git diff --binary"

    first = model.requests[0]
    assert first[0]["role"] == "system"
    assert "cluster.py:160 82% self time" in first[1]["content"]
    assert "attempt 0001" in first[1]["content"]
    last = model.requests[-1]
    assert last[2]["role"] == "assistant" and last[2].get("reasoning_content") == "thinking"
    assert last[3]["role"] == "tool"
    assert model.cache_keys == ["run-key"] * 4
    kinds = [json.loads(line)["kind"] for line in out.transcript.splitlines()]
    assert kinds[0] == "box" and kinds[-1] == "end" and "tool" in kinds and "model" in kinds


def test_history_is_rendered_and_uploaded(config: RunConfig) -> None:
    earlier = HistoryAttempt(
        ref=AttemptRef(1, 1, 0),
        base_sha=BASE_SHA,
        patch=DIFF,
        prediction=Prediction(1.5),
        rationale="tried caching",
        stop_reason=StopReason.SUBMITTED,
        usage=Usage(),
        wall_s=10.0,
        measurement=Measurement(
            noise_floor=1.0106,
            applied=True,
            tests=(
                SuiteResult("module", 5, 0, 0, 1, True),
                SuiteResult("full", 9, 1, 0, 60, False),
            ),
            base_fp="a",
            patched_fp="a",
            median_ratio=1.002,
        ),
    )
    boxes: list[FakeBox] = []

    def prep(box: FakeBox, role: str) -> None:
        prepare(box, role)
        boxes.append(box)

    factory = FakeBoxFactory(prepare=prep)
    model = FakeChatModel(
        script=[tool_call("submit", {"predicted_speedup": 1.1, "rationale": "r"})]
    )
    AgentLoopWorker(model, factory, config).attempt(make_input(config, (earlier,)))
    content = model.requests[0][1]["content"]
    assert "1 earlier attempts, 0 real speedups" in content
    assert "tried caching" in content and "full FAIL" in content and "+b" in content
    assert "median ratio vs original: 1.0020" in content and "real speedup" in content
    assert "attempt 0002" in content
    assert boxes[0].files[f"{HISTORY_DIR}/0001/patch.diff"] == DIFF.encode()
    assert f"{HISTORY_DIR}/0001/measurement.json" in boxes[0].files


def test_max_turns_cap_collects_the_diff_so_far(config: RunConfig) -> None:
    cfg = load_config(ROOT / "configs" / "t1_w1.toml")
    from dataclasses import replace

    cfg = replace(cfg, worker=replace(cfg.worker, max_turns=2))
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(script=[tool_call("read_file", {"path": "x"}) for _ in range(5)])
    out = AgentLoopWorker(model, factory, cfg).attempt(make_input(cfg))
    assert out.stop_reason == StopReason.MAX_TURNS
    assert out.turns == 2
    assert out.patch == DIFF
    assert factory.created[0].terminated


def test_max_input_tokens_cap(config: RunConfig) -> None:
    from dataclasses import replace

    cfg = replace(config, worker=replace(config.worker, max_input_tokens=150))
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(script=[tool_call("read_file", {"path": f"x{i}"}) for i in range(5)])
    out = AgentLoopWorker(model, factory, cfg).attempt(make_input(cfg))
    assert out.stop_reason == StopReason.MAX_INPUT_TOKENS
    assert out.turns == 2  # 100 tokens after turn 1, 200 after turn 2, then the cap trips


def test_repeated_identical_tool_call_ends_the_attempt(config: RunConfig) -> None:
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(script=[tool_call("read_file", {"path": "same"}) for _ in range(5)])
    out = AgentLoopWorker(model, factory, config).attempt(make_input(config))
    assert out.stop_reason == StopReason.REPEATED_TOOL_CALL
    assert out.turns == 3


def test_no_tool_call_is_nudged_once_then_ends(config: RunConfig) -> None:
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(script=[text("I think..."), text("still thinking")])
    out = AgentLoopWorker(model, factory, config).attempt(make_input(config))
    assert out.stop_reason == StopReason.NO_PROGRESS
    assert out.turns == 2
    nudges = [m for m in model.requests[1] if m["role"] == "user"]
    assert len(nudges) == 2 and "without calling a tool" in nudges[-1]["content"]


def test_model_error_after_retry_ends_the_attempt(config: RunConfig) -> None:
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(script=[ModelError("inference failed: 503")])
    out = AgentLoopWorker(model, factory, config).attempt(make_input(config))
    assert out.stop_reason == StopReason.MODEL_ERROR
    assert "503" in out.error
    assert factory.created[0].terminated


def test_box_setup_failure_terminates_and_reports(config: RunConfig) -> None:
    def broken(box: FakeBox, role: str) -> None:
        box.on("git checkout -q --detach", fail("no such commit"))

    factory = FakeBoxFactory(prepare=broken)
    model = FakeChatModel(script=[])
    out = AgentLoopWorker(model, factory, config).attempt(make_input(config))
    assert out.stop_reason == StopReason.BOX_ERROR
    assert "no such commit" in out.error
    assert out.turns == 0
    assert factory.created[0].terminated
    assert model.requests == []


def test_wrong_head_is_a_box_error(config: RunConfig) -> None:
    def wrong_tree(box: FakeBox, role: str) -> None:
        box.on("git checkout -q --detach", ok("deadbeef\n"))

    factory = FakeBoxFactory(prepare=wrong_tree)
    out = AgentLoopWorker(FakeChatModel(script=[]), factory, config).attempt(make_input(config))
    assert out.stop_reason == StopReason.BOX_ERROR
    assert "not the base" in out.error


def test_unknown_tool_is_answered_and_the_loop_continues(config: RunConfig) -> None:
    factory = FakeBoxFactory(prepare=prepare)
    model = FakeChatModel(
        script=[
            tool_call("teleport", {}),
            tool_call("submit", {"predicted_speedup": 1.0, "rationale": "nothing"}),
        ]
    )
    out = AgentLoopWorker(model, factory, config).attempt(make_input(config))
    assert out.stop_reason == StopReason.SUBMITTED
    tool_msgs = [m for m in model.requests[1] if m["role"] == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]
