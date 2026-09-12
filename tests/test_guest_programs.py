"""Runs the guest programs as real subprocesses against small local stand ins.

time_target and apply_patch are exercised end to end. count_ir needs valgrind,
which is not on the laptop, so only its parser and command shape are tested.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch.guest import count_ir, run_tests

GUEST = Path(__file__).parent.parent / "src" / "autoresearch" / "guest"


def _run(script: str, *args: str, pythonpath: Path | None = None) -> dict[str, object]:
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    proc = subprocess.run(
        [sys.executable, str(GUEST / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert lines, f"{script} printed no JSON: rc={proc.returncode} stderr={proc.stderr[-500:]}"
    data: dict[str, object] = json.loads(lines[-1])
    data["_rc"] = proc.returncode
    return data


@pytest.fixture
def fake_tree(tmp_path: Path) -> Path:
    """A tiny package that looks enough like a target: a graph builder and a call."""
    pkg = tmp_path / "tree" / "fakenx"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from fakenx.hot import compute\n\ndef make(n):\n    return list(range(n))\n"
    )
    (pkg / "hot.py").write_text("def compute(G):\n    return {i: (i * 31) % 7 for i in G}\n")
    return tmp_path / "tree"


def test_time_target_run_mode(fake_tree: Path) -> None:
    out = _run(
        "time_target.py",
        "--root",
        str(fake_tree),
        "--package",
        "fakenx",
        "--graph",
        "nx.make(500)",
        "--call",
        "nx.compute(G)",
        "--hot",
        "fakenx/hot.py",
        "--repeats",
        "3",
        "--no-counters",
    )
    assert out["_rc"] == 0
    assert out["kind"] == "run"
    assert out["repeats"] == 3
    assert isinstance(out["min_all"], float)
    assert out["min_clean"] == out["min_all"]
    assert out["n_contaminated"] == 0
    assert len(str(out["result_fp"])) == 10


def test_time_target_verify_mode_sees_the_hot_file(fake_tree: Path) -> None:
    out = _run(
        "time_target.py",
        "--root",
        str(fake_tree),
        "--package",
        "fakenx",
        "--graph",
        "nx.make(500)",
        "--call",
        "nx.compute(G)",
        "--hot",
        "fakenx/hot.py",
        "--verify",
    )
    assert out["kind"] == "verify"
    assert out["hot_executed"] is True
    assert 0.0 < float(str(out["hot_tottime_share"])) <= 1.0


def test_time_target_calls_mode(fake_tree: Path) -> None:
    out = _run(
        "time_target.py",
        "--root",
        str(fake_tree),
        "--package",
        "fakenx",
        "--graph",
        "nx.make(5)",
        "--call",
        "nx.compute(G)",
        "--hot",
        "fakenx/hot.py",
        "--calls",
        "2",
    )
    assert out["kind"] == "calls"
    assert out["calls"] == 2


def test_time_target_refuses_a_package_outside_the_tree(fake_tree: Path, tmp_path: Path) -> None:
    """Point --root at an empty directory while the package is importable via PYTHONPATH."""
    empty = tmp_path / "empty"
    empty.mkdir()
    out = _run(
        "time_target.py",
        "--root",
        str(empty),
        "--package",
        "fakenx",
        "--graph",
        "nx.make(5)",
        "--call",
        "nx.compute(G)",
        "--repeats",
        "1",
        "--no-counters",
        pythonpath=fake_tree,
    )
    assert out["_rc"] == 3
    assert "outside" in str(out["error"])


def test_canary_prints_a_minimum() -> None:
    out = _run("canary.py", "--repeats", "1")
    assert out["kind"] == "canary"
    assert out["version"] == "canary_v1"
    assert out["n"] == 2_000_000
    assert float(str(out["min_s"])) > 0


def test_run_tests_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "test_a.py").write_text(
        "def test_ok():\n    assert True\n\ndef test_bad():\n    assert False\n"
    )
    log = tmp_path / "tests.log"
    out = _run(
        "run_tests.py",
        "--root",
        str(root),
        "--target",
        "test_a.py",
        "--scope",
        "module",
        "--workers",
        "1",
        "--log",
        str(log),
    )
    assert out["kind"] == "tests"
    assert out["ok"] is False
    assert out["passed"] == 1
    assert out["failed"] == 1
    assert log.exists() and "test_bad" in log.read_text()


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "9028 passed, 61 skipped, 741 xfailed in 61.2s",
            {"passed": 9028, "skipped": 61, "xfailed": 741},
        ),
        ("1 failed, 3 passed in 0.10s", {"failed": 1, "passed": 3}),
        ("2 errors in 0.5s", {"errors": 2}),
        ("no summary here", {}),
    ],
)
def test_run_tests_summary_parser(line: str, expected: dict[str, int]) -> None:
    assert run_tests.parse_summary("junk\n" + line + "\n") == expected


def test_count_ir_parser_reads_cachegrind_summary() -> None:
    stderr = "==123== \n==123== I refs:        1,792,439,524\n==123== \n"
    assert count_ir.parse_ir(stderr) == 1_792_439_524
    assert count_ir.parse_ir("nothing") is None


def test_count_ir_command_disables_aslr_and_cache_sim() -> None:
    argv = count_ir.command(
        "python3", Path("/g/time_target.py"), "/w/a", "nx.g()", "nx.c(G)", 2, "h.py"
    )
    assert argv[0] == "setarch" and argv[2] == "-R"
    assert "--tool=cachegrind" in argv and "--cache-sim=no" in argv
    assert argv[argv.index("--calls") + 1] == "2"
    assert "--no-counters" in argv


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def test_apply_patch_worktree_lifecycle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "f.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    sha = _git(repo, "rev-parse", "HEAD").strip()
    patch = tmp_path / "p.diff"
    patch.write_text(
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    )
    wt = tmp_path / "wt"

    out = _run(
        "apply_patch.py",
        "--repo",
        str(repo),
        "--worktree",
        str(wt),
        "--commit",
        sha,
        "--patch",
        str(patch),
    )
    assert out["ok"] is True and out["applied"] is True
    assert out["changed_files"] == ["f.py"]
    assert (wt / "f.py").read_text() == "x = 2\n"
    assert (repo / "f.py").read_text() == "x = 1\n"

    bad = tmp_path / "bad.diff"
    bad.write_text(patch.read_text().replace("-x = 1", "-x = 9"))
    out = _run(
        "apply_patch.py",
        "--repo",
        str(repo),
        "--worktree",
        str(wt),
        "--commit",
        sha,
        "--patch",
        str(bad),
    )
    assert out["ok"] is False and "patch" in str(out["error"]).lower()

    out = _run("apply_patch.py", "--repo", str(repo), "--worktree", str(wt), "--remove")
    assert out["ok"] is True
    assert not wt.exists()
