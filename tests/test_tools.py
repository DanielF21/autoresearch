import json
from pathlib import Path

from autoresearch.boxes.fake_box import FakeBox, ok
from autoresearch.boxes.image import BASE_DIR, REPO_DIR
from autoresearch.config import load_config
from autoresearch.worker.tools import (
    TOOL_BY_NAME,
    TOOL_SPECS,
    ToolContext,
    execute,
)

TARGET = load_config(Path(__file__).parent.parent / "configs" / "t1_w1.toml").target


def ctx(box: FakeBox) -> ToolContext:
    return ToolContext(box=box, target=TARGET)


def test_the_tool_set_is_four_tools() -> None:
    """Reading, searching and editing are the shell's job, not a tool's.

    The three that remain each encode something the agent would otherwise have
    to rebuild correctly every attempt, and submit is how an attempt ends.
    """
    names = {s["function"]["name"] for s in TOOL_SPECS}
    assert names == set(TOOL_BY_NAME) == {"run_tests", "run_benchmark", "shell", "submit"}
    assert all(s["type"] == "function" and "parameters" in s["function"] for s in TOOL_SPECS)


def test_the_shell_description_tells_the_agent_where_it_is() -> None:
    """With no path resolving tool left, the description is the only thing that
    says where a command starts and how much output survives."""
    shell_spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "shell")
    description = shell_spec["function"]["description"]
    assert REPO_DIR in description and "12000" in description


def test_run_tests_runs_the_module_suite_and_takes_no_arguments() -> None:
    """The full suite is the referee's. A worker running it too spends its own
    budget to learn what the referee establishes on every submission anyway."""
    seen: list[str] = []
    failing = {"ok": False}

    def handler(cmd: str) -> object:
        seen.append(cmd)
        good = not failing["ok"]
        rec = {
            "ok": good,
            "passed": 5,
            "failed": 0 if good else 2,
            "errors": 0,
            "duration_s": 3.2,
            "tail": "FAILED test_x",
        }
        return ok(json.dumps(rec))

    box = FakeBox().on("run_tests.py", handler)  # type: ignore[arg-type]
    r = execute(ctx(box), "run_tests", {})
    assert r.text.startswith("module tests: PASSED: 5 passed")
    assert "--target " + TARGET.test_file in seen[-1] and "--workers 1" in seen[-1]
    assert "--scope module" in seen[-1]

    # The whole package is never the target, whatever the model passes.
    execute(ctx(box), "run_tests", {"scope": "full"})
    assert "--target networkx " not in seen[-1] and "--workers 4" not in seen[-1]
    assert "--target " + TARGET.test_file in seen[-1]

    failing["ok"] = True
    r = execute(ctx(box), "run_tests", {})
    assert r.text.startswith("module tests: FAILED") and "FAILED test_x" in r.text


def test_run_tests_offers_the_model_no_parameters() -> None:
    spec = next(s for s in TOOL_SPECS if s["function"]["name"] == "run_tests")
    assert spec["function"]["parameters"]["properties"] == {}
    assert "referee" in spec["function"]["description"]


def test_run_benchmark_alternates_order_and_reports_ratio() -> None:
    roots: list[str] = []

    def handler(cmd: str) -> object:
        root = BASE_DIR if BASE_DIR in cmd else REPO_DIR
        roots.append(root)
        return ok(json.dumps({"min_all": 1.0 if root == BASE_DIR else 0.8}))

    box = FakeBox().on("time_target.py", handler)  # type: ignore[arg-type]
    r = execute(ctx(box), "run_benchmark", {})
    assert r.text.startswith("indicative speedup 1.250x")
    assert roots == [BASE_DIR, REPO_DIR, REPO_DIR, BASE_DIR]


def test_shell_runs_in_repo_and_appends_exit_code() -> None:
    box = FakeBox().on("git status", ok("clean\n", "warn\n"))
    r = execute(ctx(box), "shell", {"cmd": "git status"})
    assert r.text == "clean\n\n[stderr]\nwarn\n\n[exit 0]"
    assert "error" in execute(ctx(box), "shell", {"cmd": "   "}).text


def test_submit_validates_and_ends() -> None:
    box = FakeBox()
    r = execute(ctx(box), "submit", {"predicted_speedup": "1.2", "rationale": "precompute sets"})
    assert r.ended and r.prediction == 1.2 and r.rationale == "precompute sets"
    assert not execute(ctx(box), "submit", {"predicted_speedup": "fast", "rationale": "x"}).ended
    assert not execute(ctx(box), "submit", {"predicted_speedup": 1.1, "rationale": ""}).ended


def test_unknown_tool_is_reported_not_raised() -> None:
    assert "unknown tool" in execute(ctx(FakeBox()), "fly", {}).text
