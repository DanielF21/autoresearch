"""The referee: one patch in, one measurement out, on one dedicated box.

The box holds the target repository at ``REPO_DIR``, checked out at the run's
base commit, and the guest programs at ``GUEST_DIR``. ``measure`` establishes
every fact it can about a patch against that base, in order: scope, apply,
module tests, full suite, one verify call of the benchmark on each tree, the
canary, six timing pairs, instruction counts, cleanup.

Nothing stops early. A step that fails records its failure in ``errors`` and the
next step still runs, with one exception: if the patched tree cannot complete a
single verify call, timing and instruction counts are skipped, because a patch
that hangs or crashes would otherwise cost up to two hours of timeouts. The
tests run regardless.

Every step that runs in the box goes through a guest program that prints one
JSON line, so the referee never parses free text. Anything that goes wrong at
the box level raises BoxError to the orchestrator, which owns the box's life.
Anything that goes wrong with the patch is a fact in the measurement.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.boxes.image import GUEST_DIR, REPO_DIR, WORK_DIR
from autoresearch.boxes.protocol import Box, BoxError, CommandResult
from autoresearch.config import RunConfig
from autoresearch.patch import changed_files, scope_violations
from autoresearch.referee import timing
from autoresearch.types import (
    IrCounts,
    Measurement,
    PairTiming,
    Provenance,
    SuiteResult,
)

GUEST_SOURCE = Path(__file__).parent.parent / "guest"
PIN_CORE = 2  # the core every Phase 1 timing ran on

TESTS_TIMEOUT = 1200
LAUNCH_TIMEOUT = 600
IR_TIMEOUT = 1800
SETUP_TIMEOUT = 300

BASE_TREE = f"{WORK_DIR}/base"
PATCHED_TREE = f"{WORK_DIR}/patched"
PATCH_FILE = f"{WORK_DIR}/attempt.diff"


class GuestError(RuntimeError):
    """A guest program did not produce a usable JSON record."""


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


class _Facts:
    """The measurement under construction. Mutable only inside ``measure``."""

    def __init__(self, noise_floor: float, box_id: str) -> None:
        self.noise_floor = noise_floor
        self.box_id = box_id
        self.applied = False
        self.apply_error = ""
        self.scope: tuple[str, ...] = ()
        self.tests: list[SuiteResult] = []
        self.base_fp = ""
        self.patched_fp = ""
        self.canary: float | None = None
        self.pairs: tuple[PairTiming, ...] = ()
        self.median: float | None = None
        self.ir: IrCounts | None = None
        self.errors: list[str] = []

    def finish(self, wall_s: float) -> Measurement:
        return Measurement(
            noise_floor=self.noise_floor,
            applied=self.applied,
            apply_error=self.apply_error,
            scope_violations=self.scope,
            tests=tuple(self.tests),
            base_fp=self.base_fp,
            patched_fp=self.patched_fp,
            canary_s=self.canary,
            pairs=self.pairs,
            speedup=self.median,
            ir=self.ir,
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
        facts = _Facts(cfg.referee.noise_floor, self._box.box_id)

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
                    self._tests(PATCHED_TREE, target.test_file, "module", workers=1)
                ),
            )
            self._step(
                facts,
                "full tests",
                lambda: facts.tests.append(
                    self._tests(PATCHED_TREE, target.hot_file.split("/")[0], "full", workers=4)
                ),
            )

            self._step(
                facts, "verify base", lambda: setattr(facts, "base_fp", self._verify(BASE_TREE))
            )
            patched_ok = self._step(
                facts,
                "verify patched",
                lambda: setattr(facts, "patched_fp", self._verify(PATCHED_TREE)),
            )
            if not patched_ok:
                facts.errors.append(
                    "timing and instruction counts skipped: patched tree cannot run"
                )
                return facts.finish(time.perf_counter() - t0)

            self._step(facts, "canary", lambda: setattr(facts, "canary", self._canary()))

            def time_it() -> None:
                pairs = self._time_pairs()
                if not timing.enough_clean(pairs, cfg.referee.min_clean_pairs):
                    pairs = self._time_pairs()
                facts.pairs = pairs
                if not timing.enough_clean(pairs, cfg.referee.min_clean_pairs):
                    facts.errors.append(
                        f"only {len(timing.clean_pairs(pairs))} clean pairs of "
                        f"{cfg.referee.pairs} after a retry; median not computed"
                    )
                    return
                facts.median = timing.median_ratio(pairs)

            self._step(facts, "timing", time_it)
            self._step(
                facts,
                "instruction counts",
                lambda: setattr(facts, "ir", self._instruction_counts()),
            )
            return facts.finish(time.perf_counter() - t0)
        finally:
            self._cleanup()

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

    def _target_args(self, tree: str) -> list[str]:
        t = self._config.target
        return ["--root", tree, "--graph", t.graph, "--call", t.call, "--hot", t.hot_file]

    def _verify(self, tree: str) -> str:
        rec = self._guest(
            "time_target.py", *self._target_args(tree), "--verify", timeout=LAUNCH_TIMEOUT
        )
        if rec.get("error"):
            raise GuestError(f"verify failed on {tree}: {rec.get('error')}")
        if not rec.get("hot_executed"):
            raise GuestError(f"hot file did not execute on {tree}")
        return str(rec.get("result_fp"))

    def _canary(self) -> float:
        rec = self._guest("canary.py", "--repeats", "5", timeout=LAUNCH_TIMEOUT)
        return float(rec.get("min_s"))

    def _launch(self, tree: str, seed: int) -> tuple[float, bool, tuple[str, ...]]:
        rec = self._guest(
            "time_target.py",
            *self._target_args(tree),
            "--repeats",
            str(self._config.referee.repeats_per_launch),
            "--label",
            tree.rsplit("/", 1)[-1],
            timeout=LAUNCH_TIMEOUT,
            env={"PYTHONHASHSEED": str(seed)},
            pin=PIN_CORE,
        )
        if rec.get("error"):
            raise GuestError(f"timing launch failed on {tree}: {rec.get('error')}")
        clean = rec.get("min_clean")
        reasons: list[str] = []
        for s in rec.get("samples", []):
            reasons.extend(s.get("reasons", []))
        if clean is None:
            return float(rec.get("min_all")), True, tuple(sorted(set(reasons)))
        return float(clean), False, ()

    def _time_pairs(self) -> tuple[PairTiming, ...]:
        cfg = self._config.referee
        out: list[PairTiming] = []
        for plan in timing.plan_pairs(cfg.pairs, cfg.hash_seeds):
            times: dict[str, float] = {}
            contaminated = False
            reasons: tuple[str, ...] = ()
            for which in plan.sequence:
                tree = BASE_TREE if which == "base" else PATCHED_TREE
                t, bad, why = self._launch(tree, plan.hash_seed)
                times[which] = t
                contaminated = contaminated or bad
                reasons = tuple(sorted(set(reasons) | set(why)))
            out.append(
                PairTiming(
                    index=plan.index,
                    order=plan.order,
                    hash_seed=plan.hash_seed,
                    base_s=times["base"],
                    patched_s=times["patched"],
                    contaminated=contaminated,
                    reasons=reasons,
                )
            )
        return tuple(out)

    def _ir_body(self, tree: str) -> int | None:
        counts: dict[int, int] = {}
        for calls in (1, 2):
            rec = self._guest(
                "count_ir.py", *self._target_args(tree), "--calls", str(calls), timeout=IR_TIMEOUT
            )
            ir = rec.get("ir")
            if ir is None:
                raise GuestError(f"cachegrind failed on {tree}: {str(rec.get('error', ''))[:300]}")
            counts[calls] = int(ir)
        return counts[2] - counts[1]

    def _instruction_counts(self) -> IrCounts | None:
        base = self._ir_body(BASE_TREE)
        patched = self._ir_body(PATCHED_TREE)
        if base is None or patched is None or base <= 0:
            return None
        return IrCounts(base=base, patched=patched)

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
