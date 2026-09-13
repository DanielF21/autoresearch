"""Builders for Scribe tests: run directories on disk, corpus records, small git repos."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from autoresearch import history
from autoresearch.scribe.corpus import CorpusPR
from autoresearch.types import (
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    Prediction,
    StopReason,
    SuiteResult,
    Usage,
    WorkerOutput,
)

FIXTURES = Path(__file__).parent / "fixtures" / "scribe"

RUN_CONFIG = """\
# the crossover sits near average degree 2, a comment no model may see
[run]
run_id = "fake_run"

[target]
repo = "{repo}"
sha = "{sha}"

[[target.inputs]]
name = "dense"
setup = "G = make(100, 0.5)"
noise_floor = 1.02

[[target.inputs]]
name = "sparse"
graph = "make(100, 0.01)"
noise_floor = 1.03
"""


def timing(name: str, speedup: float, floor: float = 1.02, base_s: float = 1.0) -> InputTiming:
    pairs = tuple(
        PairTiming(
            index=i,
            order="AB",
            hash_seed=i,
            base_s=base_s,
            patched_s=base_s / speedup,
            contaminated=False,
        )
        for i in range(6)
    )
    return InputTiming(
        name=name,
        noise_floor=floor,
        base_fp="fp",
        patched_fp="fp",
        pairs=pairs,
        speedup=speedup,
    )


def measurement(speedups: dict[str, float], tests_ok: bool = True) -> Measurement:
    return Measurement(
        applied=True,
        tests=(
            SuiteResult("module", 56, 0 if tests_ok else 1, 0, 3.0, tests_ok),
            SuiteResult("full", 9090, 0, 0, 80.0, True),
        ),
        inputs=tuple(timing(n, s) for n, s in speedups.items()),
    )


def make_run(
    root: Path,
    attempts: list[dict[str, Any]],
    *,
    repo: str = "https://example.invalid/o/r",
    sha: str = "a" * 40,
) -> Path:
    """``attempts`` items: patch, speedups (dict or None), duplicate_of, rationale, tests_ok."""
    run = root / "fake_run"
    paths = history.RunPaths(run)
    run.mkdir(parents=True)
    paths.config.write_text(RUN_CONFIG.format(repo=repo, sha=sha))
    paths.target.mkdir()
    (paths.target / "profile_inputs.txt").write_text("profile\n")
    for i, a in enumerate(attempts, start=1):
        ref = AttemptRef(i, 1, (i - 1) % 4)
        patch = a.get("patch")
        output = WorkerOutput(
            patch=patch,
            prediction=Prediction(2.0) if patch else None,
            rationale=a.get("rationale", "made it faster" if patch else ""),
            stop_reason=StopReason.SUBMITTED if patch else StopReason.BOX_ERROR,
            turns=12,
            usage=Usage(100, 50, 10, 5),
            wall_s=30.0,
            error="" if patch else "box went away",
        )
        history.write_attempt(
            paths,
            ref,
            sha,
            {},
            output,
            '{"kind": "end"}\n',
            skipped="" if patch else "no_patch",
            duplicate_of=a.get("duplicate_of", ""),
        )
        if patch and a.get("speedups") is not None:
            history.write_measurement(
                paths, ref, measurement(a["speedups"], tests_ok=a.get("tests_ok", True))
            )
    return run


def pr(number: int, title: str, body: str, *, merged_at: str = "", author: str = "dev") -> CorpusPR:
    return CorpusPR(
        number=number,
        title=title,
        body=body,
        author=author,
        labels=(),
        additions=10,
        deletions=2,
        files=("pkg/mod.py",),
        merged_at=merged_at or f"2026-09-{number % 28 + 1:02d}T00:00:00Z",
        url=f"https://example.invalid/pull/{number}",
    )


def git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(cwd),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    return r.stdout


def make_upstream(root: Path, files: dict[str, str]) -> tuple[Path, str]:
    """A small git repository with one commit. Returns its path and the commit sha."""
    repo = root / "upstream"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo, git(repo, "rev-parse", "HEAD").strip()


def diff_in(repo: Path, rel: str, new_text: str) -> str:
    """The diff that turns ``rel`` into ``new_text``, leaving the repository unchanged."""
    path = repo / rel
    old = path.read_text() if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text)
    git(repo, "add", "-N", rel)
    out = git(repo, "diff")
    if old is None:
        git(repo, "rm", "-q", "--cached", rel)
        path.unlink()
    else:
        path.write_text(old)
    return out
