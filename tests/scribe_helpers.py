"""Builders for Scribe tests: run directories on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch import history
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

RUN_CONFIG = """\
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

[worker]
model = "worker/model"
reasoning_effort = "high"
completion_window = "asap"
max_turns = 80
max_seconds = 3600
max_input_tokens = 8000000
turn_timeout = 600
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
    repo: str = "https://github.com/o/r",
    sha: str = "a" * 40,
) -> Path:
    """``attempts`` items: patch, speedups (dict or None), duplicate_of, rationale, tests_ok."""
    run = root / "fake_run"
    paths = history.RunPaths(run)
    run.mkdir(parents=True)
    paths.config.write_text(RUN_CONFIG.format(repo=repo, sha=sha))
    paths.target.mkdir()
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
