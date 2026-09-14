"""Runs the guest programs as real subprocesses against small local stand ins.

time_target, run_tests and apply_patch are exercised end to end. The stand in
package is deliberately nothing like networkx: the harness must not know what
it is timing.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch.guest import run_tests, time_target

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


def _package(pkg: Path) -> None:
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from fakepkg.hot import compute, stream\n\ndef make(n):\n    return list(range(n))\n"
    )
    (pkg / "hot.py").write_text(
        "def compute(items):\n    return {i: (i * 31) % 7 for i in items}\n\n"
        "def stream(items):\n    return (i for i in items)\n"
    )


@pytest.fixture
def fake_tree(tmp_path: Path) -> Path:
    """A tiny flat layout package: the package directory sits at the tree root."""
    _package(tmp_path / "tree" / "fakepkg")
    return tmp_path / "tree"


@pytest.fixture
def src_tree(tmp_path: Path) -> Path:
    """The same package one level down, as a src layout puts it."""
    _package(tmp_path / "tree" / "src" / "fakepkg")
    return tmp_path / "tree"


def _target(root: Path, package_root: str = ".", **overrides: str) -> list[str]:
    flags = {
        "--root": str(root),
        "--package": "fakepkg",
        "--alias": "fp",
        "--package-root": package_root,
        "--setup": "items = fp.make(500)",
        "--call": "fp.compute(items)",
        "--hot": f"{package_root}/fakepkg/hot.py" if package_root != "." else "fakepkg/hot.py",
        "--seed": "0",
    }
    flags.update(overrides)
    return [x for pair in flags.items() for x in pair]


def test_time_target_run_mode(fake_tree: Path) -> None:
    out = _run("time_target.py", *_target(fake_tree), "--repeats", "3", "--no-counters")
    assert out["_rc"] == 0
    assert out["kind"] == "run"
    assert out["repeats"] == 3
    assert isinstance(out["min_all"], float)
    assert out["min_clean"] == out["min_all"]
    assert out["n_contaminated"] == 0
    # The first call is reported apart from the best of the rest.
    assert isinstance(out["first_s"], float) and isinstance(out["warm_s"], float)
    assert out["min_all"] == min(out["first_s"], out["warm_s"])
    # The walk over the result is inside the region and reported apart.
    assert isinstance(out["walk_s"], float) and 0 <= out["walk_s"] <= out["min_all"]
    assert out["seed"] == 0
    assert len(str(out["result_fp"])) == 10
    assert isinstance(out["import_s"], float) and isinstance(out["setup_s"], float)
    assert out["python"] == list(sys.version_info[:3])
    # Process age comes from /proc, so it is a number on Linux and None elsewhere.
    assert out["fixed_s"] is None or float(str(out["fixed_s"])) > 0


def test_time_target_verify_mode_sees_the_hot_file(fake_tree: Path) -> None:
    out = _run("time_target.py", *_target(fake_tree), "--verify")
    assert out["kind"] == "verify"
    assert out["hot_executed"] is True
    assert 0.0 < float(str(out["hot_tottime_share"])) <= 1.0
    assert float(str(out["call_s"])) > 0


def test_time_target_profile_mode_times_plainly_then_profiles(fake_tree: Path) -> None:
    out = _run("time_target.py", *_target(fake_tree), "--profile", "--flat-rows", "5")
    assert out["_rc"] == 0, out
    assert out["kind"] == "profile"
    assert out["hot_executed"] is True
    assert float(str(out["call_s"])) > 0
    assert 0 < float(str(out["hot_s"])) <= float(str(out["total_s"]))
    assert 0.0 < float(str(out["hot_tottime_share"])) <= 1.0
    flat, callers = str(out["flat"]), str(out["callers"])
    # The tree's own path is stripped, so the documents read the same from any box.
    assert "fakepkg/hot.py" in flat and str(fake_tree) not in flat
    assert "tottime" in flat and "was called by" in callers
    assert len(str(out["result_fp"])) == 10


def test_time_target_imports_from_a_src_layout(src_tree: Path) -> None:
    out = _run("time_target.py", *_target(src_tree, "src"), "--repeats", "1", "--no-counters")
    assert out["_rc"] == 0, out
    assert str(out["package_file"]).startswith(str(src_tree / "src"))


def test_time_target_setup_sees_the_tree_as_root(fake_tree: Path) -> None:
    """ROOT lets an input read a file from the tree under test, whichever tree it is."""
    (fake_tree / "data.txt").write_text("7\n")
    setup = "items = fp.make(int((ROOT / 'data.txt').read_text()))"
    out = _run(
        "time_target.py",
        *_target(fake_tree, **{"--setup": setup}),
        "--repeats",
        "1",
        "--no-counters",
    )
    assert out["_rc"] == 0, out
    assert out["result_fp"] == time_target.fingerprint({i: (i * 31) % 7 for i in range(7)})


def test_time_target_refuses_a_package_outside_the_tree(fake_tree: Path, tmp_path: Path) -> None:
    """Point --root at an empty directory while the package is importable via PYTHONPATH."""
    empty = tmp_path / "empty"
    empty.mkdir()
    out = _run(
        "time_target.py",
        *_target(empty),
        "--repeats",
        "1",
        "--no-counters",
        pythonpath=fake_tree,
    )
    assert out["_rc"] == 3
    assert "outside" in str(out["error"])


def test_time_target_refuses_a_call_that_returns_an_iterator(fake_tree: Path) -> None:
    """A generator does its work when consumed, which is after the clock stops."""
    args = _target(fake_tree, **{"--call": "fp.stream(items)"})
    out = _run("time_target.py", *args, "--repeats", "1", "--no-counters")
    assert out["_rc"] == 3
    assert "iterator" in str(out["error"]) and "list()" in str(out["error"])
    out = _run("time_target.py", *args, "--verify")
    assert out["_rc"] == 3 and "iterator" in str(out["error"])


def test_time_target_hashes_the_whole_result_and_takes_no_expression(fake_tree: Path) -> None:
    """pyparsing's seeded config once fingerprinted the first 20 and last 5 tokens of
    a 10 000 token parse; a wrong middle would have matched. Every element counts."""
    args = _target(fake_tree)
    out = _run("time_target.py", *args, "--repeats", "1", "--no-counters")
    assert out["_rc"] == 0
    whole = {i: (i * 31) % 7 for i in range(500)}
    assert out["result_fp"] == time_target.fingerprint(whole)
    # The hash of a slice, the way a fingerprint expression used to reduce it.
    edges = dict(list(whole.items())[:20] + list(whole.items())[-5:])
    assert out["result_fp"] != time_target.fingerprint(edges)
    proc = subprocess.run(
        [sys.executable, str(GUEST / "time_target.py"), *args, "--fingerprint", "len(result)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2 and "--fingerprint" in proc.stderr


def test_time_target_has_no_defaults_for_the_target() -> None:
    """Every fact about the target comes from the caller; the guest knows none."""
    proc = subprocess.run(
        [sys.executable, str(GUEST / "time_target.py"), "--root", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    for flag in ("--package", "--alias", "--package-root", "--setup", "--call", "--hot", "--seed"):
        assert flag in proc.stderr


def test_time_target_binds_the_seed_and_the_instance_follows_it(fake_tree: Path) -> None:
    """SEED is the harness's handle on which instance of an input the setup builds.
    The referee times a seed the worker was never shown, so the setup has to
    read it and the result has to change with it."""
    args = _target(fake_tree, **{"--setup": "items = fp.make(500 + SEED)"})
    at_zero = _run("time_target.py", *args, "--repeats", "1", "--no-counters")
    at_seven = _run("time_target.py", *args[:-2], "--seed", "7", "--repeats", "1", "--no-counters")
    assert at_zero["_rc"] == 0 and at_seven["_rc"] == 0
    assert at_zero["seed"] == 0 and at_seven["seed"] == 7
    assert at_zero["result_fp"] != at_seven["result_fp"]
    assert at_seven["result_fp"] == time_target.fingerprint({i: (i * 31) % 7 for i in range(507)})
    verify = _run("time_target.py", *args[:-2], "--seed", "7", "--verify")
    assert verify["result_fp"] == at_seven["result_fp"]


def test_a_lazy_result_is_paid_inside_the_timed_region(fake_tree: Path) -> None:
    """pyparsing_p3_w16 from round 14 returned list subclasses that did the
    split when first read, after the clock had stopped. The walk over the
    result runs before it stops, so the deferred work is timed."""
    (fake_tree / "fakepkg" / "lazy.py").write_text(
        "import time\n\n"
        "class Lazy(list):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.done = False\n"
        "    def _fill(self):\n"
        "        if not self.done:\n"
        "            time.sleep(0.05)\n"
        "            self.extend(range(3))\n"
        "            self.done = True\n"
        "    def __iter__(self):\n"
        "        self._fill()\n"
        "        return super().__iter__()\n"
        "    def __len__(self):\n"
        "        self._fill()\n"
        "        return super().__len__()\n\n"
        "def defer(items):\n"
        "    return Lazy()\n"
    )
    args = _target(
        fake_tree, **{"--setup": "from fakepkg import lazy", "--call": "lazy.defer(None)"}
    )
    out = _run("time_target.py", *args, "--repeats", "2", "--no-counters")
    assert out["_rc"] == 0, out
    assert float(str(out["min_all"])) > 0.05
    assert float(str(out["walk_s"])) > 0.04
    assert out["result_fp"] == time_target.fingerprint([0, 1, 2])


def test_materialize_walks_what_the_fingerprint_renders() -> None:
    seen: list[str] = []

    class Arr:
        def tolist(self) -> list[int]:
            seen.append("tolist")
            return [1, 2]

    class Opaque:
        def __repr__(self) -> str:
            seen.append("repr")
            return "opaque"

    time_target.materialize({"a": [Arr(), {1, 2}], "b": (Opaque(), None, 1.5, b"x")})
    assert seen == ["tolist", "repr"]
    with pytest.raises(time_target.UnfingerprintableError, match="iterator"):
        time_target.materialize([1, (i for i in range(2))])


def test_fingerprint_depends_on_content_not_on_order_or_identity() -> None:
    fp = time_target.fingerprint
    assert fp({"b": 1, "a": 2}) == fp({"a": 2, "b": 1})
    assert fp({1: 0.1 + 0.2}) == fp({1: 0.3})
    assert fp({3, 1, 2}) == fp(frozenset([2, 3, 1]))
    assert fp([1, [2, {"x": (3.0,)}]]) == fp([1, [2, {"x": (3.0,)}]])
    assert fp([1, 2]) != fp([2, 1])  # sequences are ordered, mappings and sets are not
    assert fp("ab") != fp(["a", "b"])
    assert len({fp(None), fp(0), fp(False), fp("")}) == 4  # each renders as itself
    assert time_target.canonical(True) == "True" and time_target.canonical(1) == "1"


def test_a_value_that_is_both_mapping_and_sequence_renders_as_both() -> None:
    """pyparsing's ParseResults registers as MutableMapping and MutableSequence. Its
    named results are usually empty and its tokens are the answer; rendering the
    mapping alone hashed every parse as "{}", the same at every seed."""
    from collections.abc import MutableMapping, MutableSequence

    class Both(list):  # type: ignore[type-arg]
        def items(self) -> list[tuple[str, int]]:
            return []

    MutableMapping.register(Both)
    MutableSequence.register(Both)
    fp = time_target.fingerprint
    assert fp(Both([1, 2, 3])) != fp(Both([1, 2, 4]))
    assert fp(Both([1, 2, 3])) != fp({})
    seen: list[int] = []

    class Lazy(Both):
        def __iter__(self):  # type: ignore[no-untyped-def]
            seen.append(1)
            return super().__iter__()

    time_target.materialize(Lazy([1, 2, 3]))
    assert seen == [1]


def test_fingerprint_reads_array_like_results_through_tolist() -> None:
    class Arr:
        def tolist(self) -> list[float]:
            return [1.0, 2.5]

    assert time_target.fingerprint(Arr()) == time_target.fingerprint([1.0, 2.5])


def test_fingerprint_refuses_what_two_launches_could_never_agree_on() -> None:
    class Opaque:
        pass

    with pytest.raises(time_target.UnfingerprintableError, match="address"):
        time_target.fingerprint(Opaque())
    with pytest.raises(time_target.UnfingerprintableError, match="iterator"):
        time_target.fingerprint(iter([1]))
    with pytest.raises(time_target.UnfingerprintableError, match="iterator"):
        time_target.fingerprint({"k": (i for i in range(3))})


def test_process_age_is_a_number_on_linux_and_none_elsewhere() -> None:
    age = time_target.process_age_s()
    if Path("/proc/self/stat").exists():
        assert age is not None and 0 < age < 3600
    else:
        assert age is None


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


def test_run_tests_keeps_a_file_in_order_on_one_worker(tmp_path: Path) -> None:
    """A test that leans on what an earlier test in its file left behind must
    not pass or fail by which xdist worker drew it. pyparsing's
    testIndentedBlockClass2 did exactly that under the default distribution."""
    root = tmp_path / "proj"
    root.mkdir()
    # Each file's second test reads a mark its first test wrote into the file's
    # own directory: split across workers, the second can run first and fail.
    for n in range(6):
        (root / f"test_{n}.py").write_text(
            "import pathlib\n"
            f"MARK = pathlib.Path(__file__).parent / 'mark_{n}'\n"
            "def test_first():\n    MARK.write_text('x')\n"
            "def test_second():\n    assert MARK.exists()\n"
        )
    out = _run(
        "run_tests.py",
        "--root",
        str(root),
        "--target",
        ".",
        "--scope",
        "full",
        "--workers",
        "4",
        "--log",
        str(tmp_path / "tests.log"),
    )
    assert out["ok"] is True and out["passed"] == 12, out["tail"]


def test_run_tests_imports_the_package_from_its_package_root(
    src_tree: Path, tmp_path: Path
) -> None:
    """A src layout: the tests sit at the root, the package one level down."""
    tests = src_tree / "tests"
    tests.mkdir()
    (tests / "test_hot.py").write_text(
        "import fakepkg\n\ndef test_where():\n    assert 'src' in fakepkg.__file__\n"
    )
    out = _run(
        "run_tests.py",
        "--root",
        str(src_tree),
        "--package-root",
        "src",
        "--target",
        "tests/test_hot.py",
        "--scope",
        "module",
        "--workers",
        "1",
        "--log",
        str(tmp_path / "log"),
    )
    assert out["ok"] is True and out["passed"] == 1, out


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


def test_time_target_rebuilds_the_input_before_every_call(fake_tree: Path) -> None:
    """A result cached on the input object must never be hit: setup runs again
    before each timed call, in a fresh namespace, and only the call is timed."""
    marker = fake_tree / "setups"
    setup = (
        f"import pathlib; pathlib.Path({str(marker)!r}).open('a').write('x'); items = fp.make(500)"
    )
    out = _run(
        "time_target.py",
        *_target(fake_tree, **{"--setup": setup}),
        "--repeats",
        "4",
        "--no-counters",
    )
    assert out["_rc"] == 0 and out["repeats"] == 4
    assert marker.read_text() == "xxxx"
    one = _run(
        "time_target.py",
        *_target(fake_tree, **{"--setup": setup}),
        "--repeats",
        "1",
        "--no-counters",
    )
    assert one["_rc"] == 0 and one["warm_s"] is None
