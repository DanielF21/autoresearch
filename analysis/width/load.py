"""One row per attempt, from the run directory, with what the diff and rationale say.

Everything the referee recorded is read through ``autoresearch.history`` and
``autoresearch.types``; nothing is parsed by hand. Three columns are derived here
and are heuristics, labelled as such wherever they are printed:

- ``functions``: the functions of the base file whose lines a hunk's old range covers,
  found with ``ast`` over the base source. A hunk in a file that does not exist at the
  base commit is ``<path>:new``; one outside every function is ``<path>:module``.
- ``strategies``: keyword classes over the worker's rationale.
- ``overlap``: the share of the current record's added lines that this diff also
  adds, where the record is the best cleared attempt among those this worker saw.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch.history import (
    ATTEMPTS_DIR,
    CONFIG_FILE,
    INPUT_JSON,
    MEASUREMENT_JSON,
    OUTPUT_JSON,
    RunPaths,
    load_history,
    read_rounds,
)
from autoresearch.patch import changed_files, changed_line_count, normalised_hash
from autoresearch.types import Attempt, RoundRecord
from autoresearch.worker.prompt import outcome

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")

# Keyword classes over the rationale. A rationale may carry several. Heuristic.
STRATEGIES: dict[str, tuple[str, ...]] = {
    "bitset": ("bitmask", "bitset", "bit_count", "popcount", "bitwise", "bit mask", "bit set"),
    "matrix": ("numpy", "scipy", "einsum", "matrix", "blas", "vectoris", "vectoriz", "ndarray"),
    "cache": ("cache", "memo", "precompute", "reuse", "reused"),
    "restructure": (
        "comprehension",
        "inner loop",
        "dict lookup",
        "local variable",
        "inline",
        "generator",
        "set intersection",
        "avoid",
        "hoist",
        "sorted",
    ),
    "early_exit": ("early exit", "early return", "short circuit", "skip"),
}
BUILDS_ON = 0.5  # overlap at or above this counts as building on the record


@dataclass(frozen=True)
class Span:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class Hunk:
    path: str
    old_start: int
    old_len: int


@dataclass
class Row:
    run_id: str
    width: int
    number: int
    round: int
    worker: int
    history_n: int
    duplicate_of: str
    prompt: str  # system prompt version, from input.json; "v1" for runs before versions
    stop_reason: str
    turns: int
    wall_s: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    has_patch: bool
    applied: bool
    tests_pass: bool
    result_matches: bool | None
    per_input: dict[str, float | None]
    floors: dict[str, float]
    speedup: float | None
    worst: float | None
    regressions: tuple[str, ...]
    clears_noise: bool
    clears_seen: bool  # the flag as the referee wrote it, before the rule required every input
    complete: bool
    outcome: str
    record_seen: bool
    record: bool
    referee_wall_s: float
    files: tuple[str, ...]
    changed_lines: int
    norm_hash: str
    functions: tuple[str, ...]
    strategies: tuple[str, ...]
    rationale_chars: int
    predicted: float | None
    record_at_time: int | None
    overlap: float | None

    @property
    def outcome_class(self) -> str:
        """The outcome phrase with its detail dropped, so classes can be counted."""
        if self.outcome.startswith("no patch"):
            return "no patch"
        if self.outcome.startswith("slower on"):
            return "slower on an input"
        if self.outcome.startswith("timing failed"):
            return "timing failed"
        return self.outcome

    @property
    def builds_on_record(self) -> bool | None:
        return None if self.overlap is None else self.overlap >= BUILDS_ON


@dataclass
class Run:
    run_id: str
    width: int
    rounds_total: int
    model: str
    rounds: tuple[RoundRecord, ...]
    rows: list[Row] = field(default_factory=list)

    @property
    def rounds_done(self) -> int:
        return len(self.rounds)

    @property
    def input_names(self) -> list[str]:
        names: list[str] = []
        for r in self.rows:
            for n in r.per_input:
                if n not in names:
                    names.append(n)
        return names

    def in_round(self, n: int) -> list[Row]:
        return [r for r in self.rows if r.round == n]

    def records(self) -> list[Row]:
        return [r for r in self.rows if r.record]


BaseSource = Callable[[str], str | None]


def git_source(clone: Path, sha: str) -> BaseSource:
    """``git show sha:path`` from a clone that has the commit, cached, None if absent."""
    cache: dict[str, str | None] = {}

    def read(path: str) -> str | None:
        if path not in cache:
            r = subprocess.run(
                ["git", "-C", str(clone), "show", f"{sha}:{path}"],
                capture_output=True,
                text=True,
                check=False,
            )
            cache[path] = r.stdout if r.returncode == 0 else None
        return cache[path]

    return read


def function_spans(source: str) -> list[Span]:
    """Every function in the source, named ``Class.method`` when nested, with its lines."""
    spans: list[Span] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                spans.append(Span(name, child.lineno, child.end_lineno or child.lineno))
                walk(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    try:
        walk(ast.parse(source), "")
    except SyntaxError:
        return []
    return spans


def hunks(diff: str) -> list[Hunk]:
    out: list[Hunk] = []
    path = ""
    for line in diff.splitlines():
        m = DIFF_HEADER.match(line)
        if m:
            path = m.group(2)
            continue
        h = HUNK.match(line)
        if h:
            out.append(Hunk(path, int(h.group(1)), 1 if h.group(2) is None else int(h.group(2))))
    return out


def touched_functions(diff: str, base_source: BaseSource) -> tuple[str, ...]:
    """Innermost function of the base file for every old line a hunk covers, in order."""
    names: list[str] = []
    spans_by_path: dict[str, list[Span] | None] = {}
    for h in hunks(diff):
        if h.path not in spans_by_path:
            src = base_source(h.path)
            spans_by_path[h.path] = None if src is None else function_spans(src)
        spans = spans_by_path[h.path]
        if spans is None:
            found = [f"{h.path}:new"]
        else:
            lines = range(h.old_start, h.old_start + max(h.old_len, 1))
            found = []
            for ln in lines:
                inside = [s for s in spans if s.start <= ln <= s.end]
                name = min(inside, key=lambda s: s.end - s.start).name if inside else ""
                found.append(name or f"{h.path}:module")
        for n in found:
            if n not in names:
                names.append(n)
    return tuple(names)


def strategies(rationale: str) -> tuple[str, ...]:
    text = rationale.lower()
    found = tuple(k for k, words in STRATEGIES.items() if any(w in text for w in words))
    return found or (("other",) if text.strip() else ("none",))


def added_lines(diff: str) -> frozenset[str]:
    out: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            stripped = line[1:].strip()
            if stripped:
                out.add(stripped)
    return frozenset(out)


def _read_facts(run_dir: Path) -> tuple[str, int, int, str]:
    data = tomllib.loads((run_dir / CONFIG_FILE).read_text())
    run = data.get("run", {})
    worker = data.get("worker", {})
    return (
        str(run.get("run_id", run_dir.name)),
        int(run.get("width", 1)),
        int(run.get("rounds", 0)),
        str(worker.get("model", "")),
    )


def _row(
    run_id: str,
    width: int,
    a: Attempt,
    history_numbers: list[int],
    clears_seen: bool,
    record_seen: bool,
    record: bool,
    record_at_time: Attempt | None,
    base_source: BaseSource,
    prompt: str = "v1",
) -> Row:
    m = a.measurement
    diff = a.patch or ""
    overlap: float | None = None
    if record_at_time is not None and record_at_time.patch and a.patch:
        theirs = added_lines(record_at_time.patch)
        overlap = len(theirs & added_lines(a.patch)) / len(theirs) if theirs else None
    return Row(
        run_id=run_id,
        width=width,
        number=a.ref.number,
        round=a.ref.round,
        worker=a.ref.worker,
        history_n=len(history_numbers),
        duplicate_of=a.duplicate_of,
        prompt=prompt,
        stop_reason=str(a.stop_reason),
        turns=0,
        wall_s=a.wall_s,
        prompt_tokens=a.usage.prompt_tokens,
        cached_tokens=a.usage.cached_tokens,
        completion_tokens=a.usage.completion_tokens,
        reasoning_tokens=a.usage.reasoning_tokens,
        has_patch=a.patch is not None,
        applied=m.applied if m else False,
        tests_pass=m.tests_pass if m else False,
        result_matches=m.result_matches if m else None,
        per_input={i.name: i.speedup for i in m.inputs} if m else {},
        floors={i.name: i.noise_floor for i in m.inputs} if m else {},
        speedup=m.speedup if m else None,
        worst=m.worst_speedup if m else None,
        regressions=m.regressions if m else (),
        clears_noise=a.clears_noise,
        clears_seen=clears_seen,
        complete=complete(a),
        outcome=outcome(a),
        record_seen=record_seen,
        record=record,
        referee_wall_s=m.wall_s if m else 0.0,
        files=changed_files(diff) if diff else (),
        changed_lines=changed_line_count(diff) if diff else 0,
        norm_hash=normalised_hash(diff) if diff else "",
        functions=touched_functions(diff, base_source) if diff else (),
        strategies=strategies(a.rationale),
        rationale_chars=len(a.rationale),
        predicted=a.prediction.speedup if a.prediction else None,
        record_at_time=record_at_time.ref.number if record_at_time else None,
        overlap=overlap,
    )


def _turns(run_dir: Path, a: Attempt) -> int:
    path = run_dir / ATTEMPTS_DIR / a.ref.dirname / OUTPUT_JSON
    try:
        return int(json.loads(path.read_text()).get("turns", 0))
    except (OSError, ValueError):
        return 0


def _stored_clears(run_dir: Path, a: Attempt) -> bool:
    """``clears_noise`` as the referee wrote it, which is what the run and its workers saw.

    The property is recomputed on load, so after the rule changed to require every
    input timed, old records that cleared with an untimed input no longer do. The
    stored flag keeps the run's own view.
    """
    path = run_dir / ATTEMPTS_DIR / a.ref.dirname / MEASUREMENT_JSON
    try:
        return bool(json.loads(path.read_text()).get("clears_noise", False))
    except (OSError, ValueError):
        return False


def _input_record(run_dir: Path, a: Attempt) -> dict[str, Any]:
    path = run_dir / ATTEMPTS_DIR / a.ref.dirname / INPUT_JSON
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _history_numbers(run_dir: Path, a: Attempt) -> list[int]:
    try:
        return [int(n) for n in _input_record(run_dir, a).get("history_numbers", [])]
    except (TypeError, ValueError):
        return []


def _prompt_version(run_dir: Path, a: Attempt) -> str:
    """The system prompt version the slot ran with; runs from before versions had one."""
    value = _input_record(run_dir, a).get("prompt", "v1")
    return value if isinstance(value, str) and value else "v1"


def _speedup(a: Attempt) -> float:
    return (a.measurement.speedup or 0.0) if a.measurement else 0.0


def complete(a: Attempt) -> bool:
    """Every input produced a ratio.

    The referee leaves an input's speedup unset when too few of its timing pairs
    came back clean, and ``Measurement.speedup`` then averages the inputs that
    remain. Such an attempt can still clear the noise floor. A record here is
    only counted among complete attempts; ``record_seen`` is what the run itself
    counted, and what the workers were shown.
    """
    m = a.measurement
    return m is not None and bool(m.inputs) and all(i.speedup is not None for i in m.inputs)


def load_run(run_dir: Path, base_source: BaseSource) -> Run:
    """The run's completed rounds as rows. Attempts of a round in flight are dropped."""
    paths = RunPaths(run_dir)
    run_id, width, rounds_total, model = _read_facts(run_dir)
    rounds = read_rounds(paths)
    done = rounds[-1].round if rounds else 0
    attempts = [a for a in load_history(paths) if a.ref.round <= done]
    if len(attempts) != done * width:
        raise ValueError(
            f"{run_dir}: {len(attempts)} attempts in {done} completed rounds at width {width}"
        )
    run = Run(run_id, width, rounds_total, model, rounds)
    by_number = {a.ref.number: a for a in attempts}
    best_seen: float | None = None
    best: float | None = None
    stored = {a.ref.number: _stored_clears(run_dir, a) for a in attempts}
    for a in attempts:
        seen = _history_numbers(run_dir, a)
        cleared = [by_number[n] for n in seen if n in by_number and stored[n]]
        at_time = max(cleared, key=_speedup) if cleared else None
        speedup = a.measurement.speedup if a.measurement else None
        record_seen = (
            stored[a.ref.number]
            and speedup is not None
            and (best_seen is None or speedup > best_seen)
        )
        if record_seen:
            best_seen = speedup
        record = (
            a.clears_noise
            and complete(a)
            and speedup is not None
            and (best is None or speedup > best)
        )
        if record:
            best = speedup
        row = _row(
            run_id,
            width,
            a,
            seen,
            stored[a.ref.number],
            record_seen,
            record,
            at_time,
            base_source,
            prompt=_prompt_version(run_dir, a),
        )
        row.turns = _turns(run_dir, a)
        run.rows.append(row)
    return run
