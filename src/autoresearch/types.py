"""Records that cross a boundary between worker, referee and orchestrator.

Every record here is a frozen dataclass. Each one that is written to the run
directory has ``to_dict`` and ``from_dict`` so the on disk form is explicit and
the round trip is tested. Nothing here imports the Sail SDK.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Verdict(enum.StrEnum):
    """The referee's decision on one attempt, or why it never got one."""

    ACCEPTED = "accepted"
    REJECTED_SCOPE = "rejected_scope"
    REJECTED_APPLY = "rejected_apply"
    REJECTED_TESTS_MODULE = "rejected_tests_module"
    REJECTED_TESTS_FULL = "rejected_tests_full"
    REJECTED_BELOW_THRESHOLD = "rejected_below_threshold"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"
    NO_PATCH = "no_patch"
    UNMEASURABLE = "unmeasurable"
    FAILED = "failed"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_patch": self.patch is not None,
            "prediction": None if self.prediction is None else self.prediction.to_dict(),
            "stop_reason": str(self.stop_reason),
            "turns": self.turns,
            "usage": self.usage.to_dict(),
            "wall_s": self.wall_s,
            "error": self.error,
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
    """One back to back pair: the incumbent timed once and the patched tree timed once.

    ``ratio`` is incumbent seconds over patched seconds, so above 1 means the patch
    was faster in this pair. A contaminated pair is one where either launch tripped a
    provenance guard; it is kept in the record and excluded from the median.
    """

    index: int
    order: str
    hash_seed: int
    incumbent_s: float
    patched_s: float
    contaminated: bool
    reasons: tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        return self.incumbent_s / self.patched_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "order": self.order,
            "hash_seed": self.hash_seed,
            "incumbent_s": self.incumbent_s,
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
            incumbent_s=float(d["incumbent_s"]),
            patched_s=float(d["patched_s"]),
            contaminated=bool(d["contaminated"]),
            reasons=tuple(str(r) for r in d.get("reasons", [])),
        )


@dataclass(frozen=True)
class IrCounts:
    """Instruction counts from cachegrind for both trees, one call of the target each."""

    incumbent: int
    patched: int

    @property
    def delta_pct(self) -> float:
        return 100.0 * (self.patched - self.incumbent) / self.incumbent

    def to_dict(self) -> dict[str, Any]:
        return {"incumbent": self.incumbent, "patched": self.patched, "delta_pct": self.delta_pct}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IrCounts:
        return cls(incumbent=int(d["incumbent"]), patched=int(d["patched"]))


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
class RefereeResult:
    """The referee's full account of one attempt."""

    verdict: Verdict
    reason: str
    threshold: float
    pairs: tuple[PairTiming, ...] = ()
    median_ratio: float | None = None
    ir: IrCounts | None = None
    tests: tuple[SuiteResult, ...] = ()
    canary_s: float | None = None
    provenance: Provenance = field(default_factory=Provenance)
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "reason": self.reason,
            "threshold": self.threshold,
            "pairs": [p.to_dict() for p in self.pairs],
            "median_ratio": self.median_ratio,
            "ir": None if self.ir is None else self.ir.to_dict(),
            "tests": [t.to_dict() for t in self.tests],
            "canary_s": self.canary_s,
            "provenance": self.provenance.to_dict(),
            "wall_s": self.wall_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RefereeResult:
        ir = d.get("ir")
        median = d.get("median_ratio")
        canary = d.get("canary_s")
        return cls(
            verdict=Verdict(d["verdict"]),
            reason=str(d.get("reason", "")),
            threshold=float(d["threshold"]),
            pairs=tuple(PairTiming.from_dict(p) for p in d.get("pairs", [])),
            median_ratio=None if median is None else float(median),
            ir=None if ir is None else IrCounts.from_dict(ir),
            tests=tuple(SuiteResult.from_dict(t) for t in d.get("tests", [])),
            canary_s=None if canary is None else float(canary),
            provenance=Provenance.from_dict(d.get("provenance", {})),
            wall_s=float(d.get("wall_s", 0.0)),
        )


@dataclass(frozen=True)
class Attempt:
    """One attempt as it appears in history: what was tried and what the referee said.

    This is the unit the worker reads. ``result`` is None only while the referee has
    not run, which never happens for an attempt in a completed round.
    """

    ref: AttemptRef
    incumbent_sha: str
    patch: str | None
    prediction: Prediction | None
    rationale: str
    stop_reason: StopReason
    usage: Usage
    wall_s: float
    result: RefereeResult | None

    @property
    def verdict(self) -> Verdict | None:
        return None if self.result is None else self.result.verdict


@dataclass(frozen=True)
class RoundRecord:
    """One line of rounds.jsonl, written when a round is complete."""

    round: int
    attempt_numbers: tuple[int, ...]
    incumbent_sha_before: str
    incumbent_sha_after: str
    accepted_numbers: tuple[int, ...]
    worker_wall_s: float
    referee_wall_s: float
    usage: Usage
    errors: tuple[str, ...]
    finished_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "attempt_numbers": list(self.attempt_numbers),
            "incumbent_sha_before": self.incumbent_sha_before,
            "incumbent_sha_after": self.incumbent_sha_after,
            "accepted_numbers": list(self.accepted_numbers),
            "worker_wall_s": self.worker_wall_s,
            "referee_wall_s": self.referee_wall_s,
            "usage": self.usage.to_dict(),
            "errors": list(self.errors),
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoundRecord:
        return cls(
            round=int(d["round"]),
            attempt_numbers=tuple(int(n) for n in d["attempt_numbers"]),
            incumbent_sha_before=str(d["incumbent_sha_before"]),
            incumbent_sha_after=str(d["incumbent_sha_after"]),
            accepted_numbers=tuple(int(n) for n in d["accepted_numbers"]),
            worker_wall_s=float(d["worker_wall_s"]),
            referee_wall_s=float(d["referee_wall_s"]),
            usage=Usage.from_dict(d.get("usage", {})),
            errors=tuple(str(e) for e in d.get("errors", [])),
            finished_at=str(d["finished_at"]),
        )
