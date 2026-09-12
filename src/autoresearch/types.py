"""Records that cross a boundary between worker, referee and orchestrator.

Every record here is a frozen dataclass. Each one that is written to the run
directory has ``to_dict`` and ``from_dict`` so the on disk form is explicit and
the round trip is tested. Nothing here imports the Sail SDK.

There is no verdict anywhere. The referee returns a ``Measurement``: every fact
it could establish about a patch against the base. What those facts mean is
derived where it is needed, by the label ``clears_noise`` and by the status
command, never decided by the referee.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class StopReason(enum.StrEnum):
    """Why a worker attempt ended."""

    SUBMITTED = "submitted"
    MAX_TURNS = "max_turns"
    MAX_SECONDS = "max_seconds"
    MAX_INPUT_TOKENS = "max_input_tokens"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    NO_PROGRESS = "no_progress"
    MODEL_ERROR = "model_error"
    BOX_ERROR = "box_error"


@dataclass(frozen=True)
class AttemptRef:
    """Where an attempt sits in the run: its global number, its round, its worker slot."""

    number: int
    round: int
    worker: int

    @property
    def dirname(self) -> str:
        return f"{self.number:04d}"

    def to_dict(self) -> dict[str, Any]:
        return {"number": self.number, "round": self.round, "worker": self.worker}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AttemptRef:
        return cls(number=int(d["number"]), round=int(d["round"]), worker=int(d["worker"]))


@dataclass(frozen=True)
class Usage:
    """Token counts for one model request, or a sum of them."""

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Usage:
        return cls(
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            cached_tokens=int(d.get("cached_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            reasoning_tokens=int(d.get("reasoning_tokens", 0)),
        )


@dataclass(frozen=True)
class Prediction:
    """What the worker said before the referee measured anything."""

    speedup: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"speedup": self.speedup, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prediction:
        conf = d.get("confidence")
        return cls(speedup=float(d["speedup"]), confidence=None if conf is None else float(conf))


@dataclass(frozen=True)
class WorkerOutput:
    """Everything a worker attempt produced. ``patch`` is None when nothing was submitted."""

    patch: str | None
    prediction: Prediction | None
    rationale: str
    stop_reason: StopReason
    turns: int
    usage: Usage
    wall_s: float
    error: str = ""
    transcript: str = ""
    box_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The summary written to output.json. The transcript and patch go to their own files."""
        return {
            "has_patch": self.patch is not None,
            "prediction": None if self.prediction is None else self.prediction.to_dict(),
            "stop_reason": str(self.stop_reason),
            "turns": self.turns,
            "usage": self.usage.to_dict(),
            "wall_s": self.wall_s,
            "error": self.error,
            "box_id": self.box_id,
        }


@dataclass(frozen=True)
class SuiteResult:
    """Result of one pytest invocation."""

    scope: str
    passed: int
    failed: int
    errors: int
    duration_s: float
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_s": self.duration_s,
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SuiteResult:
        return cls(
            scope=str(d["scope"]),
            passed=int(d["passed"]),
            failed=int(d["failed"]),
            errors=int(d["errors"]),
            duration_s=float(d["duration_s"]),
            ok=bool(d["ok"]),
        )


@dataclass(frozen=True)
class PairTiming:
    """One back to back pair: the base tree timed once and the patched tree timed once.

    ``ratio`` is base seconds over patched seconds, so above 1 means the patch
    was faster in this pair. A contaminated pair is one where either launch tripped a
    provenance guard; it is kept in the record and excluded from the median.
    """

    index: int
    order: str
    hash_seed: int
    base_s: float
    patched_s: float
    contaminated: bool
    reasons: tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        return self.base_s / self.patched_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "order": self.order,
            "hash_seed": self.hash_seed,
            "base_s": self.base_s,
            "patched_s": self.patched_s,
            "ratio": self.ratio,
            "contaminated": self.contaminated,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairTiming:
        return cls(
            index=int(d["index"]),
            order=str(d["order"]),
            hash_seed=int(d["hash_seed"]),
            base_s=float(d["base_s"]),
            patched_s=float(d["patched_s"]),
            contaminated=bool(d["contaminated"]),
            reasons=tuple(str(r) for r in d.get("reasons", [])),
        )


@dataclass(frozen=True)
class IrCounts:
    """Instruction counts from cachegrind for both trees, one call of the target each."""

    base: int
    patched: int

    @property
    def delta_pct(self) -> float:
        return 100.0 * (self.patched - self.base) / self.base

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base, "patched": self.patched, "delta_pct": self.delta_pct}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IrCounts:
        return cls(base=int(d["base"]), patched=int(d["patched"]))


@dataclass(frozen=True)
class Provenance:
    """Hashes that say which machine and which software produced a measurement."""

    box_id: str = ""
    cpu_flag_hash: str = ""
    mitigation_hash: str = ""
    python_config_hash: str = ""
    package_hash: str = ""
    boot_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_id": self.box_id,
            "cpu_flag_hash": self.cpu_flag_hash,
            "mitigation_hash": self.mitigation_hash,
            "python_config_hash": self.python_config_hash,
            "package_hash": self.package_hash,
            "boot_id": self.boot_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(**{k: str(d.get(k, "")) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Measurement:
    """Every fact the referee could establish about one patch against the base.

    Steps that could not run leave their field empty or None and add a line to
    ``errors``. Nothing here is a decision. ``clears_noise`` is the one derived
    label: the patch applied, was in scope, passed both suites, computed the same
    result, and its median ratio reached the noise floor.
    """

    noise_floor: float
    applied: bool = False
    apply_error: str = ""
    scope_violations: tuple[str, ...] = ()
    tests: tuple[SuiteResult, ...] = ()
    base_fp: str = ""
    patched_fp: str = ""
    canary_s: float | None = None
    pairs: tuple[PairTiming, ...] = ()
    median_ratio: float | None = None
    ir: IrCounts | None = None
    errors: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    wall_s: float = 0.0

    @property
    def tests_pass(self) -> bool:
        scopes = {t.scope: t.ok for t in self.tests}
        return bool(scopes) and all(scopes.values()) and "module" in scopes and "full" in scopes

    @property
    def result_matches(self) -> bool | None:
        if not self.base_fp or not self.patched_fp:
            return None
        return self.base_fp == self.patched_fp

    @property
    def clears_noise(self) -> bool:
        return (
            self.applied
            and not self.scope_violations
            and self.tests_pass
            and self.result_matches is True
            and self.median_ratio is not None
            and self.median_ratio >= self.noise_floor
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_floor": self.noise_floor,
            "applied": self.applied,
            "apply_error": self.apply_error,
            "scope_violations": list(self.scope_violations),
            "tests": [t.to_dict() for t in self.tests],
            "tests_pass": self.tests_pass,
            "base_fp": self.base_fp,
            "patched_fp": self.patched_fp,
            "result_matches": self.result_matches,
            "canary_s": self.canary_s,
            "pairs": [p.to_dict() for p in self.pairs],
            "median_ratio": self.median_ratio,
            "clears_noise": self.clears_noise,
            "ir": None if self.ir is None else self.ir.to_dict(),
            "errors": list(self.errors),
            "provenance": self.provenance.to_dict(),
            "wall_s": self.wall_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Measurement:
        ir = d.get("ir")
        median = d.get("median_ratio")
        canary = d.get("canary_s")
        return cls(
            noise_floor=float(d["noise_floor"]),
            applied=bool(d.get("applied", False)),
            apply_error=str(d.get("apply_error", "")),
            scope_violations=tuple(str(v) for v in d.get("scope_violations", [])),
            tests=tuple(SuiteResult.from_dict(t) for t in d.get("tests", [])),
            base_fp=str(d.get("base_fp", "")),
            patched_fp=str(d.get("patched_fp", "")),
            canary_s=None if canary is None else float(canary),
            pairs=tuple(PairTiming.from_dict(p) for p in d.get("pairs", [])),
            median_ratio=None if median is None else float(median),
            ir=None if ir is None else IrCounts.from_dict(ir),
            errors=tuple(str(e) for e in d.get("errors", [])),
            provenance=Provenance.from_dict(d.get("provenance", {})),
            wall_s=float(d.get("wall_s", 0.0)),
        )


@dataclass(frozen=True)
class Attempt:
    """One attempt as it appears in history: what was tried and what was measured.

    This is the unit the worker reads. ``measurement`` is None only when the
    attempt produced no patch, in which case ``skipped`` says so, or while the
    referee has not run, which never happens for an attempt in a completed round.
    ``duplicate_of`` names an earlier attempt with the same normalised diff; the
    attempt is still measured in full.
    """

    ref: AttemptRef
    base_sha: str
    patch: str | None
    prediction: Prediction | None
    rationale: str
    stop_reason: StopReason
    usage: Usage
    wall_s: float
    measurement: Measurement | None
    skipped: str = ""
    duplicate_of: str = ""

    @property
    def clears_noise(self) -> bool:
        return self.measurement is not None and self.measurement.clears_noise


@dataclass(frozen=True)
class RoundRecord:
    """One line of rounds.jsonl, written when a round is complete."""

    round: int
    attempt_numbers: tuple[int, ...]
    base_sha: str
    measured_numbers: tuple[int, ...]
    clears_noise_numbers: tuple[int, ...]
    best_ratio_so_far: float | None
    worker_wall_s: float
    referee_wall_s: float
    usage: Usage
    errors: tuple[str, ...]
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "attempt_numbers": list(self.attempt_numbers),
            "base_sha": self.base_sha,
            "measured_numbers": list(self.measured_numbers),
            "clears_noise_numbers": list(self.clears_noise_numbers),
            "best_ratio_so_far": self.best_ratio_so_far,
            "worker_wall_s": self.worker_wall_s,
            "referee_wall_s": self.referee_wall_s,
            "usage": self.usage.to_dict(),
            "errors": list(self.errors),
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoundRecord:
        best = d.get("best_ratio_so_far")
        return cls(
            round=int(d["round"]),
            attempt_numbers=tuple(int(n) for n in d["attempt_numbers"]),
            base_sha=str(d["base_sha"]),
            measured_numbers=tuple(int(n) for n in d.get("measured_numbers", [])),
            clears_noise_numbers=tuple(int(n) for n in d.get("clears_noise_numbers", [])),
            best_ratio_so_far=None if best is None else float(best),
            worker_wall_s=float(d["worker_wall_s"]),
            referee_wall_s=float(d["referee_wall_s"]),
            usage=Usage.from_dict(d.get("usage", {})),
            errors=tuple(str(e) for e in d.get("errors", [])),
            finished_at=str(d["finished_at"]),
        )
