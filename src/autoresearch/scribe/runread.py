"""The one place the Scribe reads a run directory.

Nothing else in the Scribe imports ``history`` or ``Measurement``. Every attempt
becomes a ``Candidate``: the patch, what the worker said, and the measured facts
flattened into plain fields. When the run record changes shape, this module is
the one that changes.

The run's ``config.toml`` is read here for the repo, the sha and each input's
setup, and for nothing else. Its comments are never passed on: in t1_w4c they
describe the dense and sparse crossover, which is the answer a reviewer is meant
to find from the diff.
"""

from __future__ import annotations

import json
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

from autoresearch import history
from autoresearch.types import Attempt, InputTiming


@dataclass(frozen=True)
class RunTarget:
    run_id: str
    repo: str
    sha: str
    setups: dict[str, str]


@dataclass(frozen=True)
class InputRow:
    name: str
    setup: str
    noise_floor: float
    speedup: float | None
    pairs: int
    clean_pairs: int
    base_median_s: float | None
    patched_median_s: float | None
    improves: bool
    regresses: bool
    result_matches: bool | None


@dataclass(frozen=True)
class SuiteRow:
    scope: str
    passed: int
    failed: int
    errors: int
    ok: bool


@dataclass(frozen=True)
class Candidate:
    """One attempt, as the Scribe sees it."""

    run_id: str
    number: int
    patch: str | None
    rationale: str
    predicted_speedup: float | None
    stop_reason: str
    error: str
    turns: int
    skipped: str
    duplicate_of: str
    measured: bool
    applied: bool
    scope_violations: tuple[str, ...]
    tests: tuple[SuiteRow, ...]
    tests_pass: bool
    result_matches: bool | None
    regressions: tuple[str, ...]
    clears_noise: bool
    speedup: float | None
    worst_speedup: float | None
    inputs: tuple[InputRow, ...]

    @property
    def dirname(self) -> str:
        return f"{self.number:04d}"


def read_target(run_dir: Path) -> RunTarget:
    """Repo, sha and setups, from either config shape.

    The older shape binds each input's value to ``G`` through a ``graph`` key; the
    current one states ``setup`` statements. Both are rendered as setup text.
    """
    data = tomllib.loads((run_dir / history.CONFIG_FILE).read_text())
    target = data["target"]
    setups: dict[str, str] = {}
    for item in target.get("inputs", []):
        if "setup" in item:
            setups[str(item["name"])] = str(item["setup"])
        elif "graph" in item:
            setups[str(item["name"])] = f"G = {item['graph']}"
    return RunTarget(
        run_id=str(data.get("run", {}).get("run_id", run_dir.name)),
        repo=str(target["repo"]),
        sha=str(target["sha"]),
        setups=setups,
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _input_row(timing: InputTiming, setups: dict[str, str]) -> InputRow:
    clean = [p for p in timing.pairs if not p.contaminated]
    return InputRow(
        name=timing.name,
        setup=setups.get(timing.name, ""),
        noise_floor=timing.noise_floor,
        speedup=timing.speedup,
        pairs=len(timing.pairs),
        clean_pairs=len(clean),
        base_median_s=_median([p.base_s for p in clean]),
        patched_median_s=_median([p.patched_s for p in clean]),
        improves=timing.improves,
        regresses=timing.regresses,
        result_matches=timing.result_matches,
    )


def _candidate(
    run_id: str, attempt: Attempt, attempt_dir: Path, setups: dict[str, str]
) -> Candidate:
    output = json.loads((attempt_dir / history.OUTPUT_JSON).read_text())
    m = attempt.measurement
    return Candidate(
        run_id=run_id,
        number=attempt.ref.number,
        patch=attempt.patch,
        rationale=attempt.rationale,
        predicted_speedup=None if attempt.prediction is None else attempt.prediction.speedup,
        stop_reason=str(attempt.stop_reason),
        error=str(output.get("error", "")),
        turns=int(output.get("turns", 0)),
        skipped=attempt.skipped,
        duplicate_of=attempt.duplicate_of,
        measured=m is not None,
        applied=bool(m and m.applied),
        scope_violations=() if m is None else m.scope_violations,
        tests=()
        if m is None
        else tuple(SuiteRow(t.scope, t.passed, t.failed, t.errors, t.ok) for t in m.tests),
        tests_pass=bool(m and m.tests_pass),
        result_matches=None if m is None else m.result_matches,
        regressions=() if m is None else m.regressions,
        clears_noise=attempt.clears_noise,
        speedup=None if m is None else m.speedup,
        worst_speedup=None if m is None else m.worst_speedup,
        inputs=() if m is None else tuple(_input_row(i, setups) for i in m.inputs),
    )


def load_candidates(run_dir: Path) -> tuple[Candidate, ...]:
    paths = history.RunPaths(run_dir)
    target = read_target(run_dir)
    return tuple(
        _candidate(target.run_id, a, paths.attempt(a.ref), target.setups)
        for a in history.load_history(paths)
    )


def run_head(run_dir: Path) -> str:
    """The run directory's own git HEAD, read without running git. Empty if absent."""
    head = run_dir / ".git" / "HEAD"
    if not head.exists():
        return ""
    ref = head.read_text().strip()
    if not ref.startswith("ref: "):
        return ref
    target = run_dir / ".git" / ref[5:]
    return target.read_text().strip() if target.exists() else ""


def is_locked(run_dir: Path) -> bool:
    return history.RunPaths(run_dir).lock.exists()


def attempt_files(run_dir: Path, number: int, *, transcripts: bool) -> dict[str, Path]:
    """The run files a model may read about one attempt, keyed by run relative path.

    Never ``config.toml``, never ``input.json`` or ``output.json``: what a model
    needs from those is already in the Candidate it is shown.
    """
    d = history.RunPaths(run_dir).attempts / f"{number:04d}"
    names = [history.PATCH_DIFF, history.MEASUREMENT_JSON, history.RATIONALE_MD]
    if transcripts:
        names.append(history.TRANSCRIPT_JSONL)
    return {
        f"{history.ATTEMPTS_DIR}/{number:04d}/{name}": d / name
        for name in names
        if (d / name).exists()
    }


def profile_files(run_dir: Path) -> dict[str, Path]:
    target = history.RunPaths(run_dir).target
    if not target.exists():
        return {}
    return {f"{history.TARGET_DIR}/{p.name}": p for p in sorted(target.glob("profile_*.txt"))}
