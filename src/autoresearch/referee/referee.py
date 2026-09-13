"""The referee: one patch in, one measurement out, on one dedicated box.

The box holds the target repository at ``REPO_DIR``, checked out at the run's
base commit, and the guest programs at ``GUEST_DIR``. ``measure`` establishes
every fact it can about a patch against that base, in order: scope, apply,
module tests, full suite, one verify call of every input on each tree, the
canary, six timing pairs per input, cleanup.

Everything the referee knows about the target it reads from ``TargetSpec``:
what to import and under what name, where in the tree it imports from, each
input's setup, the call, the fingerprint rule, and which pytest paths are the
module's tests and the whole suite. Nothing about any particular target is in
this file.

The target is timed on a set of inputs, not one. Each input carries its own
noise floor, since a one millisecond call's ratio scatters about twice as wide
as a 1.4 second one; artifacts/generality.md has the measurements that led to
both. An input with no floor yet cannot be measured, only calibrated, and
``measure`` refuses it by name.

Nothing stops early. A step that fails records its failure in ``errors`` and the
next step still runs, with one exception, per input: an input the patched tree
cannot complete a verify call on is not timed, because a patch that hangs or
crashes would otherwise cost twelve launch timeouts for that input. The other
inputs are still timed and the tests run regardless.

Every step that runs in the box goes through a guest program that prints one
JSON line, so the referee never parses free text. One launch times one input on
one tree. Anything that goes wrong at the box level raises BoxError to the
orchestrator, which owns the box's life. Anything that goes wrong with the patch
is a fact in the measurement.
"""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from autoresearch.boxes.image import GUEST_DIR, REPO_DIR, WORK_DIR
from autoresearch.boxes.protocol import Box, BoxError, CommandResult
from autoresearch.config import BenchmarkInput, RunConfig, TargetSpec
from autoresearch.patch import changed_files, scope_violations
from autoresearch.referee import timing
from autoresearch.types import (
    InputTiming,
    Measurement,
    PairTiming,
    Provenance,
    SuiteResult,
)

GUEST_SOURCE = Path(__file__).parent.parent / "guest"
PIN_CORE = 2  # the core every Phase 1 timing ran on

TESTS_TIMEOUT = 1200
LAUNCH_TIMEOUT = 600
SETUP_TIMEOUT = 300

BASE_TREE = f"{WORK_DIR}/base"
PATCHED_TREE = f"{WORK_DIR}/patched"
PATCH_FILE = f"{WORK_DIR}/attempt.diff"


class GuestError(RuntimeError):
    """A guest program did not produce a usable JSON record."""


@dataclass(frozen=True)
class _Launch:
    """One timing launch: its best sample, whether any sample was contaminated,
    why, and what the process paid before its first call."""

    seconds: float
    contaminated: bool
    reasons: tuple[str, ...]
    fixed_s: float | None


@dataclass(frozen=True)
class InputSurvey:
    """One input as the base tree computes it, from two verify launches.

    ``error`` is set when the input could not run at all, and then nothing
    else is meaningful. ``fingerprints`` are the two launches' results, which
    agree for a deterministic call and are the whole of the determinism check.
    """

    name: str
    error: str = ""
    hot_executed: bool = False
    hot_share: float = 0.0
    call_s: float = 0.0
    import_s: float = 0.0
    setup_s: float = 0.0
    fixed_s: float | None = None
    python: str = ""
    fingerprints: tuple[str, str] = ("", "")

    @property
    def deterministic(self) -> bool:
        return not self.error and self.fingerprints[0] == self.fingerprints[1]


@dataclass(frozen=True)
class Survey:
    inputs: tuple[InputSurvey, ...]
    tests: tuple[SuiteResult, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputProfile:
    """One input on the base tree: plain call time, and where cProfile saw it go."""

    name: str
    setup: str
    call_s: float
    hot_s: float
    share: float
    flat: str
    callers: str


@dataclass(frozen=True)
class GuestRecord:
    data: dict[str, Any]
    raw: CommandResult

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def guest_command(script: str, *args: str, pin: int | None = None) -> str:
    """The shell line that runs one guest program, optionally pinned to a core."""
    quoted = " ".join(shlex.quote(a) for a in args)
    prefix = f"taskset -c {pin} " if pin is not None else ""
    return f"cd {GUEST_DIR} && {prefix}python3 {script} {quoted}"


def target_args(target: TargetSpec, tree: str, spec: BenchmarkInput) -> list[str]:
    """Every fact time_target.py needs about the target, as its flags.

    One place, used by the referee and by the worker's benchmark tool, so the
    two cannot drift into timing different things.
    """
    args = [
        "--root",
        tree,
        "--package",
        target.package,
        "--alias",
        target.alias,
        "--package-root",
        target.package_root,
        "--setup",
        spec.setup,
        "--call",
        target.call,
        "--hot",
        target.hot_file,
    ]
    if target.fingerprint:
        args += ["--fingerprint", target.fingerprint]
    return args


class _InputFacts:
    """One input's timing under construction, in the order the referee learns it."""

    def __init__(self, spec: BenchmarkInput) -> None:
        if spec.noise_floor is None:
            raise ValueError(f"input {spec.name!r} has no noise floor; calibrate it first")
        self.spec = spec
        self.noise_floor: float = spec.noise_floor
        self.base_fp = ""
        self.patched_fp = ""
        self.pairs: tuple[PairTiming, ...] = ()
        self.speedup: float | None = None
        self.retried = False
        self.errors: list[str] = []

    def finish(self) -> InputTiming:
        return InputTiming(
            name=self.spec.name,
            noise_floor=self.noise_floor,
            base_fp=self.base_fp,
            patched_fp=self.patched_fp,
            pairs=self.pairs,
            speedup=self.speedup,
            retried=self.retried,
            errors=tuple(self.errors),
        )


class _Facts:
    """The measurement under construction. Mutable only inside ``measure``."""

    def __init__(self, inputs: tuple[BenchmarkInput, ...], box_id: str) -> None:
        self.box_id = box_id
        self.applied = False
        self.apply_error = ""
        self.scope: tuple[str, ...] = ()
        self.tests: list[SuiteResult] = []
        self.canary: float | None = None
        self.inputs = [_InputFacts(spec) for spec in inputs]
        self.errors: list[str] = []

    def finish(self, wall_s: float) -> Measurement:
        return Measurement(
            applied=self.applied,
            apply_error=self.apply_error,
            scope_violations=self.scope,
            tests=tuple(self.tests),
            canary_s=self.canary,
            inputs=tuple(i.finish() for i in self.inputs),
            errors=tuple(self.errors),
            provenance=Provenance(box_id=self.box_id),
            wall_s=wall_s,
        )


class Referee:
    def __init__(self, box: Box, config: RunConfig) -> None:
        self._box = box
        self._config = config
        self._broken = ""

    @property
    def box(self) -> Box:
        return self._box

    @property
    def broken(self) -> str:
        """Non empty when cleanup failed and the box should be rebuilt before reuse."""
        return self._broken

    # ----- setup ---------------------------------------------------------------------

    def setup(self) -> None:
        """Upload guest programs and confirm the repo is at the run's base commit."""
        self._box.upload_dir(GUEST_SOURCE, GUEST_DIR)
        base = self._config.target.sha
        r = self._box.run(
            f"mkdir -p {WORK_DIR} && cd {REPO_DIR} && git checkout -q --detach {base}"
            " && git rev-parse HEAD",
            timeout=SETUP_TIMEOUT,
        )
        if not r.ok:
            raise BoxError(f"referee box has no repo at {REPO_DIR}: {r.stderr[-500:]}")
        head = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if head != base:
            raise BoxError(f"referee box is at {head}, not the base {base}")

    # ----- measuring -----------------------------------------------------------------

    def measure(self, patch: str) -> Measurement:
        t0 = time.perf_counter()
        cfg = self._config
        target = cfg.target
        if target.uncalibrated:
            raise ValueError(
                "cannot measure with uncalibrated inputs: " + ", ".join(target.uncalibrated)
            )
        facts = _Facts(target.inputs, self._box.box_id)

        files = changed_files(patch)
        facts.scope = tuple(
            f"{v.path}: {v.reason}" for v in scope_violations(files, target.allow, target.deny)
        )
        if not files:
            facts.apply_error = "diff touches no files"
            return facts.finish(time.perf_counter() - t0)

        try:
            self._box.write(PATCH_FILE, patch.encode())
            self._worktree(BASE_TREE, target.sha, patch_file=None)
            applied = self._worktree(PATCHED_TREE, target.sha, patch_file=PATCH_FILE)
            facts.applied = bool(applied.get("ok"))
            if not facts.applied:
                facts.apply_error = str(applied.get("error", ""))[:500]
                return facts.finish(time.perf_counter() - t0)

            self._step(
                facts,
                "module tests",
                lambda: facts.tests.append(
                    self._tests(PATCHED_TREE, target.tests.module, "module", workers=1)
                ),
            )
            self._step(
                facts,
                "full tests",
                lambda: facts.tests.append(
                    self._tests(PATCHED_TREE, target.tests.full, "full", workers=4)
                ),
            )

            # Verify every input on both trees before timing any of them, so a
            # patch that cannot run at all is found in one cheap call per input
            # rather than in sixty launch timeouts.
            for inp in facts.inputs:
                self._verify_input(facts, inp)

            timeable = [i for i in facts.inputs if i.base_fp and i.patched_fp]
            if not timeable:
                return facts.finish(time.perf_counter() - t0)

            # The canary reads the machine's health so a timing can be read in
            # context, so it is worth a launch only if something will be timed.
            self._step(facts, "canary", lambda: setattr(facts, "canary", self._canary()))

            # partial, not a lambda: a lambda would close over the loop variable
            # rather than bind it, which is correct only because _step calls it
            # at once. Binding here does not depend on that.
            for inp in timeable:
                self._step(facts, f"timing {inp.spec.name}", partial(self._time_input, inp))
            return facts.finish(time.perf_counter() - t0)
        finally:
            self._cleanup()

    def null_pairs(
        self,
        rounds: int,
        only: tuple[str, ...] = (),
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, tuple[PairTiming, ...]]:
        """Pairs from two worktrees of the same commit: what no change looks like.

        This is where a noise floor comes from. The floor has to be the spread of
        the referee's own statistic when the patch does nothing, so this runs the
        same pairing plan, hash seeds, core pin and repeats per launch that
        ``measure`` runs. A separate timing loop written for calibration could
        drift from the one being calibrated and produce a floor for a procedure
        nobody uses.

        Both trees are the base commit, so the null is exact rather than a patch
        assumed to be inert. No tests and no canary: neither says anything about
        timing spread, and the launches are what the time is worth spending on.

        ``rounds`` is how many times each input's full set of pairs is run, so an
        input comes back with ``rounds * referee.pairs`` of them.
        """
        target = self._config.target
        specs = [i for i in target.inputs if not only or i.name in only]
        missing = set(only) - {i.name for i in specs}
        if missing:
            raise ValueError(f"no such input: {', '.join(sorted(missing))}")
        out: dict[str, list[PairTiming]] = {i.name: [] for i in specs}
        try:
            self._worktree(BASE_TREE, target.sha, patch_file=None)
            self._worktree(PATCHED_TREE, target.sha, patch_file=None)
            for spec in specs:
                self._verify(BASE_TREE, spec)
                self._verify(PATCHED_TREE, spec)
            for _ in range(rounds):
                for spec in specs:
                    out[spec.name].extend(self._time_pairs(spec))
                    if progress is not None:
                        progress(spec.name, len(out[spec.name]))
        finally:
            self._cleanup()
        return {name: tuple(pairs) for name, pairs in out.items()}

    def survey(self) -> Survey:
        """Every fact about the target on the base tree alone, with no patch.

        What the check command reads: each input verified twice, so a result
        that is not deterministic shows as two fingerprints, and both suites
        run once. Nothing here needs a noise floor, so an uncalibrated target
        can be surveyed, which is the point: this is how a candidate target is
        accepted or refused before anything is calibrated or run.
        """
        target = self._config.target
        inputs: list[InputSurvey] = []
        suites: list[SuiteResult] = []
        errors: list[str] = []
        try:
            self._worktree(BASE_TREE, target.sha, patch_file=None)
            for spec in target.inputs:
                inputs.append(self._survey_input(spec))
            for scope, path, workers in (
                ("module", target.tests.module, 1),
                ("full", target.tests.full, 4),
            ):
                try:
                    suites.append(self._tests(BASE_TREE, path, scope, workers))
                except GuestError as e:
                    errors.append(f"{scope} tests: {e}")
        finally:
            self._cleanup()
        return Survey(inputs=tuple(inputs), tests=tuple(suites), errors=tuple(errors))

    def profile(self) -> tuple[InputProfile, ...]:
        """Every input profiled on the base tree, in config order. No patch, no floor.

        What the profile documents are made from, so a worker reads times taken on
        a referee box rather than on whatever machine wrote the config. An input
        that cannot run raises GuestError: documents missing an input would tell
        the worker the set is smaller than the one it is graded on.
        """
        target = self._config.target
        out: list[InputProfile] = []
        try:
            self._worktree(BASE_TREE, target.sha, patch_file=None)
            for spec in target.inputs:
                rec = self._guest(
                    "time_target.py",
                    *self._target_args(BASE_TREE, spec),
                    "--profile",
                    timeout=LAUNCH_TIMEOUT,
                )
                if rec.get("error"):
                    raise GuestError(f"profile failed for {spec.name}: {rec.get('error')}")
                out.append(
                    InputProfile(
                        name=spec.name,
                        setup=spec.setup,
                        call_s=float(rec.get("call_s", 0.0)),
                        hot_s=float(rec.get("hot_s", 0.0)),
                        share=float(rec.get("hot_tottime_share", 0.0)),
                        flat=str(rec.get("flat", "")),
                        callers=str(rec.get("callers", "")),
                    )
                )
        finally:
            self._cleanup()
        return tuple(out)

    def _survey_input(self, spec: BenchmarkInput) -> InputSurvey:
        try:
            first = self._verify_record(BASE_TREE, spec)
            second = self._verify_record(BASE_TREE, spec)
        except GuestError as e:
            return InputSurvey(name=spec.name, error=str(e))
        return InputSurvey(
            name=spec.name,
            hot_executed=bool(first.get("hot_executed")),
            hot_share=float(first.get("hot_tottime_share", 0.0)),
            call_s=float(first.get("call_s", 0.0)),
            import_s=float(first.get("import_s", 0.0)),
            setup_s=float(first.get("setup_s", 0.0)),
            fixed_s=None if first.get("fixed_s") is None else float(first.get("fixed_s")),
            python=".".join(str(v) for v in first.get("python", [])),
            fingerprints=(str(first.get("result_fp")), str(second.get("result_fp"))),
        )

    def _verify_input(self, facts: _Facts, inp: _InputFacts) -> None:
        """One verify call per tree. A failure on either leaves this input untimed."""
        self._step(
            facts,
            f"verify base {inp.spec.name}",
            lambda: setattr(inp, "base_fp", self._verify(BASE_TREE, inp.spec)),
        )
        ok = self._step(
            facts,
            f"verify patched {inp.spec.name}",
            lambda: setattr(inp, "patched_fp", self._verify(PATCHED_TREE, inp.spec)),
        )
        if not ok:
            inp.errors.append("timing skipped: patched tree cannot run this input")

    def _time_input(self, inp: _InputFacts) -> None:
        """Six pairs, retried once if too few came back clean.

        The retry replaces the first pass rather than adding to it, so only the
        second pass is recorded. ``retried`` says that happened: without it a
        retry is invisible in the record and shows up only as unexplained wall
        clock, which is what attempt 0009 of t1_w4b turned out to be.
        """
        cfg = self._config.referee
        pairs = self._time_pairs(inp.spec)
        if not timing.enough_clean(pairs, cfg.min_clean_pairs):
            inp.retried = True
            pairs = self._time_pairs(inp.spec)
        inp.pairs = pairs
        if not timing.enough_clean(pairs, cfg.min_clean_pairs):
            inp.errors.append(
                f"only {len(timing.clean_pairs(pairs))} clean pairs of "
                f"{cfg.pairs} after a retry; median not computed"
            )
            return
        inp.speedup = timing.median_ratio(pairs)

    def _step(self, facts: _Facts, name: str, run: Any) -> bool:
        """Run one step. A guest failure is a recorded fact; a box failure propagates."""
        try:
            run()
            return True
        except GuestError as e:
            facts.errors.append(f"{name}: {e}")
            return False

    # ----- guest calls ---------------------------------------------------------------

    def _guest(
        self,
        script: str,
        *args: str,
        timeout: int,
        env: dict[str, str] | None = None,
        pin: int | None = None,
    ) -> GuestRecord:
        r = self._box.run(guest_command(script, *args, pin=pin), timeout=timeout, env=env)
        line = r.last_json_line()
        if line is None:
            raise GuestError(
                f"{script} printed no JSON (rc {r.exit_code}"
                f"{', timed out' if r.timed_out else ''}): {r.stderr[-800:]}"
            )
        return GuestRecord(json.loads(line), r)

    def _worktree(self, path: str, commit: str, patch_file: str | None) -> GuestRecord:
        args = ["--repo", REPO_DIR, "--worktree", path, "--commit", commit]
        if patch_file:
            args += ["--patch", patch_file]
        rec = self._guest("apply_patch.py", *args, timeout=SETUP_TIMEOUT)
        if patch_file is None and not rec.get("ok"):
            raise BoxError(f"could not create the base worktree: {rec.get('error')}")
        return rec

    def _tests(self, tree: str, target: str, scope: str, workers: int) -> SuiteResult:
        log = f"{WORK_DIR}/tests_{scope}.log"
        rec = self._guest(
            "run_tests.py",
            "--root",
            tree,
            "--package-root",
            self._config.target.package_root,
            "--target",
            target,
            "--scope",
            scope,
            "--workers",
            str(workers),
            "--log",
            log,
            "--timeout",
            str(TESTS_TIMEOUT - 60),
            timeout=TESTS_TIMEOUT,
        )
        return SuiteResult(
            scope=scope,
            passed=int(rec.get("passed", 0)),
            failed=int(rec.get("failed", 0)),
            errors=int(rec.get("errors", 0)),
            duration_s=float(rec.get("duration_s", 0.0)),
            ok=bool(rec.get("ok", False)),
        )

    def _target_args(self, tree: str, spec: BenchmarkInput) -> list[str]:
        return target_args(self._config.target, tree, spec)

    def _verify_record(self, tree: str, spec: BenchmarkInput) -> GuestRecord:
        rec = self._guest(
            "time_target.py", *self._target_args(tree, spec), "--verify", timeout=LAUNCH_TIMEOUT
        )
        if rec.get("error"):
            raise GuestError(f"verify failed on {tree} for {spec.name}: {rec.get('error')}")
        return rec

    def _verify(self, tree: str, spec: BenchmarkInput) -> str:
        rec = self._verify_record(tree, spec)
        if not rec.get("hot_executed"):
            raise GuestError(f"hot file did not execute on {tree} for {spec.name}")
        return str(rec.get("result_fp"))

    def _canary(self) -> float:
        rec = self._guest("canary.py", "--repeats", "5", timeout=LAUNCH_TIMEOUT)
        return float(rec.get("min_s"))

    def _launch(self, tree: str, spec: BenchmarkInput, seed: int) -> _Launch:
        rec = self._guest(
            "time_target.py",
            *self._target_args(tree, spec),
            "--repeats",
            str(self._config.referee.repeats_per_launch),
            "--label",
            f"{spec.name}/{tree.rsplit('/', 1)[-1]}",
            timeout=LAUNCH_TIMEOUT,
            env={"PYTHONHASHSEED": str(seed)},
            pin=PIN_CORE,
        )
        if rec.get("error"):
            raise GuestError(f"timing launch failed on {tree} for {spec.name}: {rec.get('error')}")
        fixed = rec.get("fixed_s")
        fixed_s = None if fixed is None else float(fixed)
        clean = rec.get("min_clean")
        reasons: list[str] = []
        for s in rec.get("samples", []):
            reasons.extend(s.get("reasons", []))
        if clean is None:
            return _Launch(float(rec.get("min_all")), True, tuple(sorted(set(reasons))), fixed_s)
        return _Launch(float(clean), False, (), fixed_s)

    def _time_pairs(self, spec: BenchmarkInput) -> tuple[PairTiming, ...]:
        """One input's pairs. The two launches of a pair are back to back so
        machine drift affects both and cancels, which is why every input gets
        its own pairs rather than sharing a launch."""
        cfg = self._config.referee
        out: list[PairTiming] = []
        for plan in timing.plan_pairs(cfg.pairs, cfg.hash_seeds):
            launches: dict[str, _Launch] = {}
            contaminated = False
            reasons: tuple[str, ...] = ()
            for which in plan.sequence:
                tree = BASE_TREE if which == "base" else PATCHED_TREE
                launch = self._launch(tree, spec, plan.hash_seed)
                launches[which] = launch
                contaminated = contaminated or launch.contaminated
                reasons = tuple(sorted(set(reasons) | set(launch.reasons)))
            out.append(
                PairTiming(
                    index=plan.index,
                    order=plan.order,
                    hash_seed=plan.hash_seed,
                    base_s=launches["base"].seconds,
                    patched_s=launches["patched"].seconds,
                    contaminated=contaminated,
                    reasons=reasons,
                    base_fixed_s=launches["base"].fixed_s,
                    patched_fixed_s=launches["patched"].fixed_s,
                )
            )
        return tuple(out)

    def _cleanup(self) -> None:
        """Remove both worktrees. A failure marks the referee broken rather than
        raising, so a measurement already made is not lost; the orchestrator
        checks ``broken`` and rebuilds the box before the next attempt."""
        for path in (BASE_TREE, PATCHED_TREE):
            try:
                r = self._box.run(
                    guest_command(
                        "apply_patch.py", "--repo", REPO_DIR, "--worktree", path, "--remove"
                    ),
                    timeout=SETUP_TIMEOUT,
                )
                line = r.last_json_line()
                removed = bool(json.loads(line).get("ok")) if line else False
                if not r.ok or not removed:
                    self._broken = f"cleanup of {path} failed: {r.stderr[-300:]}"
            except BoxError as e:
                self._broken = f"cleanup of {path} raised: {e}"
