"""The local repository cache and the confined read only tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.scribe import tools
from autoresearch.scribe.repo import PatchApplyError, RepoCache, SubprocessGit, read_sources
from tests.scribe_helpers import diff_in, git, make_upstream

MOD = "".join(f"line {i}\n" for i in range(1, 11))


class CountingGit(SubprocessGit):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, args: list[str], cwd: Path, timeout: float = 300.0) -> str:
        self.calls.append(args)
        return super().run(args, cwd, timeout)


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str]:
    return make_upstream(tmp_path, {"pkg/mod.py": MOD, "README.md": "hello\n"})


def test_trees_are_checked_out_once_and_patched_trees_hold_the_patch(
    tmp_path: Path, upstream: tuple[Path, str]
) -> None:
    repo, sha = upstream
    patch = diff_in(repo, "pkg/mod.py", MOD.replace("line 3\n", "line three\n"))
    g = CountingGit()
    cache = RepoCache(tmp_path / "cache", g)
    base = cache.tree(f"file://{repo}", sha)
    patched = cache.tree(f"file://{repo}", sha, patch)
    assert (base / "pkg/mod.py").read_text() == MOD
    assert "line three" in (patched / "pkg/mod.py").read_text()
    assert read_sources(patched, ("pkg/mod.py", "missing.py")).keys() == {"pkg/mod.py"}

    before = len(g.calls)
    assert cache.tree(f"file://{repo}", sha, patch) == patched
    assert len(g.calls) - before == 1  # the commit check only; no checkout, no apply
    assert not any(c[:2] == ["worktree", "remove"] for c in g.calls)


def test_a_patch_that_does_not_apply_leaves_no_tree(
    tmp_path: Path, upstream: tuple[Path, str]
) -> None:
    repo, sha = upstream
    bad = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-nope\n+x\n"
    cache = RepoCache(tmp_path / "cache")
    with pytest.raises(PatchApplyError):
        cache.tree(f"file://{repo}", sha, bad)
    trees = tmp_path / "cache" / "trees" / sha[:12]
    assert [p.name for p in trees.iterdir() if p.is_dir()] == []
    # And the cache still works afterwards, which needs the worktree list pruned.
    assert (cache.tree(f"file://{repo}", sha) / "README.md").exists()


def make_ctx(tmp_path: Path, upstream: tuple[Path, str]) -> tuple[tools.ToolContext, Path]:
    repo, sha = upstream
    cache = RepoCache(tmp_path / "cache")
    base = cache.tree(f"file://{repo}", sha)
    run_file = tmp_path / "patch.diff"
    run_file.write_text("diff text with needle\n")
    secret = tmp_path / "config.toml"
    secret.write_text("the answer\n")
    roots = tools.Roots(
        dirs={"base": base},
        virtual={"run": {"attempts/0001/patch.diff": run_file}},
    )
    ctx = tools.ToolContext(
        roots=roots, git=SubprocessGit(), clone=cache.clone(f"file://{repo}", sha), base_sha=sha
    )
    return ctx, base


def call(ctx: tools.ToolContext, name: str, **args: object) -> str:
    return tools.execute(tools.READ_TOOLS, ctx, name, dict(args)).text


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("base:../outside.txt", "escapes"),
        ("base:/etc/passwd", "absolute"),
        ("elsewhere:x", "unknown root"),
        ("base:-rf", "may not start with '-'"),
        ("base:pkg\x00mod.py", "NUL"),
        ("pkg/mod.py", "root:relative/path"),
        ("run:config.toml", "not available"),
        ("run:../config.toml", "escapes"),
    ],
)
def test_paths_outside_their_roots_are_refused(
    tmp_path: Path, upstream: tuple[Path, str], path: str, message: str
) -> None:
    ctx, _ = make_ctx(tmp_path, upstream)
    out = call(ctx, "read_file", path=path)
    assert out.startswith("error:") and message in out


def test_a_symlink_out_of_a_tree_is_refused(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    ctx, base = make_ctx(tmp_path, upstream)
    (base / "leak.txt").symlink_to(tmp_path / "config.toml")
    assert "escapes" in call(ctx, "read_file", path="base:leak.txt")
    assert "the answer" not in call(ctx, "grep", pattern="answer", path="base:")


def test_read_list_and_grep_inside_roots(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    ctx, _ = make_ctx(tmp_path, upstream)
    out = call(ctx, "read_file", path="base:pkg/mod.py", start_line=3, end_line=4)
    assert "     3  line 3" in out and "     4  line 4" in out and "line 5" not in out
    assert "lines 3 to 4 of 10" in out

    listing = call(ctx, "list_dir", path="base:", depth=2)
    assert "base:pkg/" in listing and "base:pkg/mod.py" in listing and ".git" not in listing
    assert call(ctx, "list_dir", path="run:") == "run:attempts/0001/patch.diff"

    hits = call(ctx, "grep", pattern=r"line 1\d?$", path="base:", glob="pkg/*.py")
    assert "base:pkg/mod.py:1: line 1" in hits and "base:pkg/mod.py:10: line 10" in hits
    assert "run:attempts/0001/patch.diff:1:" in call(ctx, "grep", pattern="needle", path="run:")
    assert "invalid regular expression" in call(ctx, "grep", pattern="(", path="base:")


def test_git_tools_take_no_flags_and_stay_behind_the_base(
    tmp_path: Path, upstream: tuple[Path, str]
) -> None:
    repo, sha = upstream
    (repo / "later.txt").write_text("after the base\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "after base")
    later = git(repo, "rev-parse", "HEAD").strip()

    ctx, _ = make_ctx(tmp_path, upstream)
    assert "base" in call(ctx, "git_log", path="pkg/mod.py")
    assert "after base" not in call(ctx, "git_log")
    target = tmp_path / "written.txt"
    assert call(ctx, "git_log", path=f"--output={target}").startswith("error:")
    assert not target.exists()
    assert "hex" in call(ctx, "git_show", rev="--output=x")
    assert "not an ancestor" in call(ctx, "git_show", rev=later)
    assert "pkg/mod.py" in call(ctx, "git_show", rev=sha[:10])


def test_unknown_tools_and_bad_arguments_come_back_as_text(
    tmp_path: Path, upstream: tuple[Path, str]
) -> None:
    ctx, _ = make_ctx(tmp_path, upstream)
    assert "unknown tool" in call(ctx, "shell", command="ls")
    assert "must be an integer" in call(ctx, "read_file", path="base:README.md", start_line="x")
    assert tools.clip("a" * 50, 20).count("clipped") == 1
