"""Every attempt in a run as a Candidate, and the filter that decides which ones a model sees.

The filter is code. A candidate is kept only when the referee's record is clean on
every count, and each one dropped carries its reasons. This is the one Scribe module
that reads the run record, so when that record changes shape, this is what changes.
"""

from __future__ import annotations

import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

from autoresearch import history
from autoresearch.patch import changed_line_count
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
    result_matches: bool | None


@dataclass(frozen=True)
class SuiteRow:
    scope: str
    passed: int
    failed: int
    errors: int


@dataclass(frozen=True)
class Candidate:
    number: int
    patch: str | None
    rationale: str
    stop_reason: str
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
        result_matches=timing.result_matches,
    )


def _candidate(attempt: Attempt, setups: dict[str, str]) -> Candidate:
    m = attempt.measurement
    return Candidate(
        number=attempt.ref.number,
        patch=attempt.patch,
        rationale=attempt.rationale,
        stop_reason=str(attempt.stop_reason),
        skipped=attempt.skipped,
        duplicate_of=attempt.duplicate_of,
        measured=m is not None,
        applied=bool(m and m.applied),
        scope_violations=() if m is None else m.scope_violations,
        tests=()
        if m is None
        else tuple(SuiteRow(t.scope, t.passed, t.failed, t.errors) for t in m.tests),
        tests_pass=bool(m and m.tests_pass),
        result_matches=None if m is None else m.result_matches,
        regressions=() if m is None else m.regressions,
        clears_noise=attempt.clears_noise,
        speedup=None if m is None else m.speedup,
        worst_speedup=None if m is None else m.worst_speedup,
        inputs=() if m is None else tuple(_input_row(i, setups) for i in m.inputs),
    )


def load_candidates(run_dir: Path) -> tuple[Candidate, ...]:
    target = read_target(run_dir)
    return tuple(
        _candidate(a, target.setups) for a in history.load_history(history.RunPaths(run_dir))
    )


# ----- the filter -----------------------------------------------------------------------


def filter_reasons(c: Candidate) -> tuple[str, ...]:
    """Why a candidate is dropped. Empty when it is kept."""
    reasons: list[str] = []
    if not c.patch:
        reasons.append(f"no patch ({c.skipped or c.stop_reason})")
    elif not c.measured:
        reasons.append("not measured")
    else:
        if not c.applied:
            reasons.append("did not apply")
        if c.scope_violations:
            reasons.append("scope violations: " + ", ".join(c.scope_violations))
        if not c.tests_pass:
            reasons.append("tests did not pass")
        if c.result_matches is not True:
            reasons.append("results do not match on every input")
        if c.regressions:
            reasons.append("regresses on: " + ", ".join(c.regressions))
        if not c.clears_noise:
            reasons.append("does not clear the noise floor")
    if c.duplicate_of:
        reasons.append(f"duplicate of attempt {c.duplicate_of}")
    return tuple(reasons)


def filter_candidates(
    candidates: tuple[Candidate, ...],
) -> tuple[list[Candidate], dict[int, tuple[str, ...]]]:
    """The kept candidates, and every dropped attempt number with its reasons."""
    kept: list[Candidate] = []
    excluded: dict[int, tuple[str, ...]] = {}
    for c in candidates:
        reasons = filter_reasons(c)
        if reasons:
            excluded[c.number] = reasons
        else:
            kept.append(c)
    return kept, excluded


# Chosen, not measured. Eight candidates made a 44k character pick message on t1_w4d;
# a long run can keep hundreds, which would not fit one prompt.
MAX_CANDIDATES = 12


def shortlist(kept: list[Candidate]) -> tuple[list[Candidate], dict[int, tuple[str, ...]]]:
    """At most ``MAX_CANDIDATES``: the fastest half, then the smallest diffs of the rest.

    Speed alone would drop the small change a maintainer merges; size alone would drop
    the change worth reading. The shortlist keeps attempt order, and every candidate left
    out comes back with its reason.
    """
    if len(kept) <= MAX_CANDIDATES:
        return list(kept), {}
    half = MAX_CANDIDATES // 2
    fastest = sorted(kept, key=lambda c: (-(c.speedup or 0.0), c.number))[:half]
    chosen = {c.number for c in fastest}
    rest = [c for c in kept if c.number not in chosen]
    smallest = sorted(rest, key=lambda c: (changed_line_count(c.patch or ""), c.number))
    chosen |= {c.number for c in smallest[: MAX_CANDIDATES - half]}
    reason = (
        f"not shortlisted: not among the {half} fastest or the "
        f"{MAX_CANDIDATES - half} smallest diffs of the rest",
    )
    return (
        [c for c in kept if c.number in chosen],
        {c.number: reason for c in kept if c.number not in chosen},
    )


# ----- what the model reads about a candidate -------------------------------------------


def fmt_ratio(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}x"


def fmt_seconds(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 1000:.1f} ms" if v < 1 else f"{v:.3f} s"


def inputs_table(c: Candidate) -> str:
    rows = [
        "| input | setup | speedup | noise floor | clean pairs | base median | patched median | same result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in c.inputs:
        same = {True: "yes", False: "NO", None: "unknown"}[i.result_matches]
        rows.append(
            f"| {i.name} | `{i.setup}` | {fmt_ratio(i.speedup)} | {i.noise_floor:.4f} | "
            f"{i.clean_pairs} of {i.pairs} | {fmt_seconds(i.base_median_s)} | "
            f"{fmt_seconds(i.patched_median_s)} | {same} |"
        )
    rows.append("")
    rows.append(
        f"Geometric mean over {len(c.inputs)} inputs: {fmt_ratio(c.speedup)}. "
        f"Worst input: {fmt_ratio(c.worst_speedup)}. "
        "A speedup is base seconds over patched seconds, the median of the clean pairs."
    )
    return "\n".join(rows)


def tests_line(c: Candidate) -> str:
    if not c.tests:
        return "No test results recorded."
    parts = [
        f"{t.scope} suite: {t.passed} passed, {t.failed} failed, {t.errors} errors" for t in c.tests
    ]
    return "; ".join(parts) + "."


def describe(c: Candidate) -> str:
    return "\n".join(
        [
            f"## Attempt {c.number}",
            "",
            f"{changed_line_count(c.patch or '')} lines changed.",
            "",
            "```diff",
            (c.patch or "").rstrip(),
            "```",
            "",
            "### Measurements",
            inputs_table(c),
            "",
            tests_line(c),
            "",
            "### The worker's rationale",
            c.rationale.strip() or "(none)",
        ]
    )
