import json
from pathlib import Path

from autoresearch.boxes.fake_box import FakeBox, fail, ok
from autoresearch.boxes.image import REPO_DIR
from autoresearch.config import load_config
from autoresearch.worker.tools import (
    BASE_DIR,
    HISTORY_DIR,
    TOOL_BY_NAME,
    TOOL_SPECS,
    ToolContext,
    execute,
)

TARGET = load_config(Path(__file__).parent.parent / "configs" / "t1_w1.toml").target


def ctx(box: FakeBox) -> ToolContext:
    return ToolContext(box=box, target=TARGET)


def test_specs_are_openai_function_tools() -> None:
    names = {s["function"]["name"] for s in TOOL_SPECS}
    assert (
        names
        == set(TOOL_BY_NAME)
        == {
            "read_file",
            "search",
            "edit_file",
            "write_file",
            "run_tests",
            "run_benchmark",
            "shell",
            "submit",
        }
    )
    assert all(s["type"] == "function" and "parameters" in s["function"] for s in TOOL_SPECS)


def test_read_file_numbers_lines_and_resolves_relative_paths() -> None:
    box = FakeBox().on(
        "awk", lambda cmd: ok("     1\tx = 1\n") if f"{REPO_DIR}/a/b.py" in cmd else fail()
    )
    r = execute(ctx(box), "read_file", {"path": "a/b.py"})
    assert r.text == "     1\tx = 1\n"
    assert "NR>=1 && NR<=300" in box.commands[-1]
    r = execute(ctx(box), "read_file", {"path": "a/b.py", "start": 10, "end": 20})
    assert "NR>=10 && NR<=20" in box.commands[-1]


def test_paths_may_not_escape_the_repo() -> None:
    box = FakeBox()
    assert "error" in execute(ctx(box), "read_file", {"path": "../etc/passwd"}).text
    assert "error" in execute(ctx(box), "read_file", {"path": "/etc/passwd"}).text
    assert box.commands == []
    box.on("awk", ok("     1\thi\n"))
    assert "hi" in execute(ctx(box), "read_file", {"path": f"{HISTORY_DIR}/0001/patch.diff"}).text


def test_search_defaults_to_repo_and_reports_no_matches() -> None:
    box = FakeBox().on("grep", lambda cmd: ok("") if REPO_DIR in cmd else fail())
    r = execute(ctx(box), "search", {"pattern": "def clustering"})
    assert r.text == "(no matches)"
    assert "-e 'def clustering'" in box.commands[-1]


def test_edit_file_requires_exactly_one_match() -> None:
    box = FakeBox()
    path = f"{REPO_DIR}/x.py"
    box.write(path, b"a = 1\nb = 1\n")
    r = execute(ctx(box), "edit_file", {"path": "x.py", "old": "= 1", "new": "= 2"})
    assert "occurs 2 times" in r.text
    assert box.read(path) == b"a = 1\nb = 1\n"
    r = execute(ctx(box), "edit_file", {"path": "x.py", "old": "a = 1", "new": "a = 2"})
    assert r.text == f"edited {path}"
    assert box.read(path) == b"a = 2\nb = 1\n"
    assert (
        "error"
        in execute(ctx(box), "edit_file", {"path": "missing.py", "old": "a", "new": "b"}).text
    )


def test_write_file() -> None:
    box = FakeBox()
    r = execute(ctx(box), "write_file", {"path": "new.py", "content": "print(1)\n"})
    assert "wrote 9 characters" in r.text
    assert box.read(f"{REPO_DIR}/new.py") == b"print(1)\n"


def test_run_tests_module_and_full_scopes() -> None:
    seen: list[str] = []

    def handler(cmd: str) -> object:
        seen.append(cmd)
        good = "--scope module" in cmd
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
    r = execute(ctx(box), "run_tests", {"scope": "module"})
    assert r.text.startswith("module tests: PASSED: 5 passed")
    assert "--target " + TARGET.test_file in seen[-1] and "--workers 1" in seen[-1]
    r = execute(ctx(box), "run_tests", {"scope": "full"})
    assert r.text.startswith("full tests: FAILED") and "FAILED test_x" in r.text
    assert "--target networkx " in seen[-1] and "--workers 4" in seen[-1]
    assert "error" in execute(ctx(box), "run_tests", {"scope": "nope"}).text


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
