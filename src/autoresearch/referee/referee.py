"""The referee: one patch in, one verdict out, on one dedicated box.

The box holds the target repository at ``REPO_DIR`` and the guest programs at
``GUEST_DIR``. ``sync_incumbent`` brings the box's copy of the incumbent up to
date with the orchestrator's, checked by tree hash. ``judge`` then does, in
order: scope check, worktrees, module tests, full tests, result check, canary,
timing pairs, instruction counts, verdict, cleanup.

Every step that runs in the box goes through a guest program that prints one
JSON line, so the referee never parses free text. Anything that goes wrong at
the box level raises BoxError to the orchestrator, which owns the box's life.
Anything that goes wrong with the patch is a verdict, not an exception.
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
    PairTiming,
    Provenance,
    RefereeResult,
    SuiteResult,
    Verdict,
)

GUEST_SOURCE = Path(__file__).parent.parent / "guest"
PIN_CORE = 2  # the core every Phase 1 timing ran on

TESTS_TIMEOUT = 1200
LAUNCH_TIMEOUT = 600
IR_TIMEOUT = 1800
SETUP_TIMEOUT = 300

INCUMBENT_TREE = f"{WORK_DIR}/incumbent"
PATCHED_TREE = f"{WORK_DIR}/patched"
PATCH_FILE = f"{WORK_DIR}/attempt.diff"
STACK_FILE = f"{WORK_DIR}/incumbent.diff"


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


class Referee:
    def __init__(self, box: Box, config: RunConfig) -> None:
        self._box = box
        self._config = config
        self._box_head = ""
        self._broken = ""

    @property
    def box(self) -> Box:
        return self._box

    @property
    def broken(self) -> str:
        """Non empty when cleanup failed and the box should be rebuilt before reuse."""
        return self._broken

    # ----- setup and incumbent sync -------------------------------------------------

    def setup(self) -> None:
        """Upload guest programs and confirm the repo is where the image put it."""
        self._box.upload_dir(GUEST_SOURCE, GUEST_DIR)
        r = self._box.run(
            f"mkdir -p {WORK_DIR} && cd {REPO_DIR} && git rev-parse HEAD", timeout=SETUP_TIMEOUT
        )
        if not r.ok:
            raise BoxError(f"referee box has no repo at {REPO_DIR}: {r.stderr[-500:]}")
        self._box_head = r.stdout.strip()

    def sync_incumbent(self, base_sha: str, stack_diff: str, expected_tree: str) -> str:
        """Make the box's repo match the orchestrator's incumbent. Returns the box's head.

        The box replays the accepted patches as one diff on top of the pinned
        commit, then its tree hash must equal the orchestrator's. Commit ids
        differ between the two, tree hashes do not.
        """
        self._box.write(STACK_FILE, stack_diff.encode())
        apply = (
            f"git apply --index {STACK_FILE} && git commit -q -m incumbent"
            if stack_diff.strip()
            else "true"
        )
        r = self._box.run(
            f"cd {REPO_DIR} && git checkout -q --detach {base_sha} && {apply}"
            f" && git rev-parse HEAD && git rev-parse 'HEAD^{{tree}}'",
            timeout=SETUP_TIMEOUT,
        )
        if not r.ok:
            raise BoxError(f"incumbent sync failed: {r.stderr[-500:]}")
        lines = r.stdout.split()
        head, tree = lines[-2], lines[-1]
        if tree != expected_tree:
            raise BoxError(f"incumbent tree mismatch: box {tree} vs orchestrator {expected_tree}")
        self._box_head = head
        return head

    # ----- judging -------------------------------------------------------------------

    def judge(self, incumbent_sha: str, patch: str) -> RefereeResult:
        t0 = time.perf_counter()
        cfg = self._config
        target = cfg.target
        tests: list[SuiteResult] = []

        def done(verdict: Verdict, reason: str, **extra: Any) -> RefereeResult:
            return RefereeResult(
                verdict=verdict,
                reason=reason,
                threshold=cfg.referee.threshold,
                tests=tuple(tests),
                provenance=Provenance(box_id=self._box.box_id),
                wall_s=time.perf_counter() - t0,
                **extra,
            )

        files = changed_files(patch)
        if not files:
            return done(Verdict.REJECTED_APPLY, "diff touches no files")
        violations = scope_violations(files, target.allow, target.deny)
        if violations:
            detail = "; ".join(f"{v.path}: {v.reason}" for v in violations)
            return done(Verdict.REJECTED_SCOPE, detail)

        try:
            self._box.write(PATCH_FILE, patch.encode())
            base = self._box_head or incumbent_sha
            self._worktree(INCUMBENT_TREE, base, patch_file=None)
            applied = self._worktree(PATCHED_TREE, base, patch_file=PATCH_FILE)
            if not applied.get("ok"):
                return done(Verdict.REJECTED_APPLY, str(applied.get("error", ""))[:500])

            module = self._tests(PATCHED_TREE, target.test_file, "module", workers=1)
            tests.append(module)
            if not module.ok:
                return done(
                    Verdict.REJECTED_TESTS_MODULE, f"{module.failed} failed, {module.errors} errors"
                )
            full = self._tests(PATCHED_TREE, target.hot_file.split("/")[0], "full", workers=4)
            tests.append(full)
            if not full.ok:
                return done(
                    Verdict.REJECTED_TESTS_FULL, f"{full.failed} failed, {full.errors} errors"
                )

            fp_inc = self._verify(INCUMBENT_TREE)
            fp_pat = self._verify(PATCHED_TREE)
            if fp_inc != fp_pat:
                return done(Verdict.FAILED, f"benchmark result differs: {fp_inc} vs {fp_pat}")

            canary = self._canary()
            pairs = self._time_pairs()
            if not timing.enough_clean(pairs, cfg.referee.min_clean_pairs):
                pairs = self._time_pairs()
            if not timing.enough_clean(pairs, cfg.referee.min_clean_pairs):
                return done(
                    Verdict.UNMEASURABLE,
                    f"only {len(timing.clean_pairs(pairs))} clean pairs of {cfg.referee.pairs} after a retry",
                    pairs=pairs,
                    canary_s=canary,
                )

            median = timing.median_ratio(pairs)
            ir = self._instruction_counts()
            if timing.is_speedup(median, cfg.referee.threshold):
                verdict, reason = (
                    Verdict.ACCEPTED,
                    f"median {median:.4f} >= {cfg.referee.threshold}",
                )
            else:
                verdict, reason = (
                    Verdict.REJECTED_BELOW_THRESHOLD,
                    f"median {median:.4f} < {cfg.referee.threshold}",
                )
            return done(verdict, reason, pairs=pairs, median_ratio=median, ir=ir, canary_s=canary)
        finally:
            self._cleanup()

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
            raise GuestError(f"{script} printed no JSON (rc {r.exit_code}): {r.stderr[-800:]}")
        return GuestRecord(json.loads(line), r)

    def _worktree(self, path: str, commit: str, patch_file: str | None) -> GuestRecord:
        args = ["--repo", REPO_DIR, "--worktree", path, "--commit", commit]
        if patch_file:
            args += ["--patch", patch_file]
        rec = self._guest("apply_patch.py", *args, timeout=SETUP_TIMEOUT)
        if patch_file is None and not rec.get("ok"):
            raise BoxError(f"could not create the incumbent worktree: {rec.get('error')}")
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
            raise BoxError(f"verify failed on {tree}: {rec.get('error')}")
        if not rec.get("hot_executed"):
            raise BoxError(f"hot file did not execute on {tree}")
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
            raise BoxError(f"timing launch failed on {tree}: {rec.get('error')}")
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
                tree = INCUMBENT_TREE if which == "incumbent" else PATCHED_TREE
                t, bad, why = self._launch(tree, plan.hash_seed)
                times[which] = t
                contaminated = contaminated or bad
                reasons = tuple(sorted(set(reasons) | set(why)))
            out.append(
                PairTiming(
                    index=plan.index,
                    order=plan.order,
                    hash_seed=plan.hash_seed,
                    incumbent_s=times["incumbent"],
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
                return None
            counts[calls] = int(ir)
        return counts[2] - counts[1]

    def _instruction_counts(self) -> IrCounts | None:
        inc = self._ir_body(INCUMBENT_TREE)
        pat = self._ir_body(PATCHED_TREE)
        if inc is None or pat is None or inc <= 0:
            return None
        return IrCounts(incumbent=inc, patched=pat)

    def _cleanup(self) -> None:
        """Remove both worktrees. A failure marks the referee broken rather than
        raising, so a verdict already reached is not lost; the orchestrator
        checks ``broken`` and rebuilds the box before the next attempt."""
        for path in (INCUMBENT_TREE, PATCHED_TREE):
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
