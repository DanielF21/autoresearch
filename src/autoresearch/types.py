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
import math
from dataclasses import dataclass, field
from typing import Any

from autoresearch.referee.thresholds import PILOT_NOISE_FLOOR


class StopReason(enum.StrEnum):
    """Why a worker attempt ended."""

    SUBMITTED = "submitted"
    LAST_TURN = "last_turn"  # submitted, on the turn the harness announced as the last
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

    ``base_fixed_s`` and ``patched_fixed_s`` are what each launch paid before its
    first timed call: interpreter start, the import and the input's setup, as
    the guest measured its own process age. None in records written before the
    guest reported it, and on a machine with no /proc.
    """

    index: int
    order: str
    hash_seed: int
    base_s: float
    patched_s: float
    contaminated: bool
    reasons: tuple[str, ...] = ()
    base_fixed_s: float | None = None
    patched_fixed_s: float | None = None

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
            "base_fixed_s": self.base_fixed_s,
            "patched_fixed_s": self.patched_fixed_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairTiming:
        base_fixed = d.get("base_fixed_s")
        patched_fixed = d.get("patched_fixed_s")
        return cls(
            index=int(d["index"]),
            order=str(d["order"]),
            hash_seed=int(d["hash_seed"]),
            base_s=float(d["base_s"]),
            patched_s=float(d["patched_s"]),
            contaminated=bool(d["contaminated"]),
            reasons=tuple(str(r) for r in d.get("reasons", [])),
            base_fixed_s=None if base_fixed is None else float(base_fixed),
            patched_fixed_s=None if patched_fixed is None else float(patched_fixed),
        )


@dataclass(frozen=True)
class InputTiming:
    """Everything the referee established about one patch on one benchmark input.

    Each input carries its own noise floor, because the floor is a property of
    how long the call takes rather than of the patch. ``regresses`` is the
    mirror of ``improves``: as far below 1 as the floor is above it, so the test
    is symmetric and a slowdown has to clear the same bar a speedup does.

    ``speedup`` is None when too few pairs came back clean after every retry.
    Such an input is ``untimed``, and a measurement with one never clears the
    noise floor: the mean over the inputs that remain says nothing about the
    input that is missing, and the missing ones are the cheap ones where a
    patch gains least.
    """

    name: str
    noise_floor: float
    base_fp: str = ""
    patched_fp: str = ""
    pairs: tuple[PairTiming, ...] = ()
    speedup: float | None = None
    retried: bool = False
    errors: tuple[str, ...] = ()
    retries: int = 0

    @property
    def untimed(self) -> bool:
        return self.speedup is None

    @property
    def result_matches(self) -> bool | None:
        if not self.base_fp or not self.patched_fp:
            return None
        return self.base_fp == self.patched_fp

    @property
    def improves(self) -> bool:
        return self.speedup is not None and self.speedup >= self.noise_floor

    @property
    def regresses(self) -> bool:
        return self.speedup is not None and self.speedup < 1.0 / self.noise_floor

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "noise_floor": self.noise_floor,
            "base_fp": self.base_fp,
            "patched_fp": self.patched_fp,
            "result_matches": self.result_matches,
            "pairs": [p.to_dict() for p in self.pairs],
            "speedup": self.speedup,
            "improves": self.improves,
            "regresses": self.regresses,
            "untimed": self.untimed,
            "retried": self.retried,
            "retries": self.retries,
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InputTiming:
        speedup = d.get("speedup")
        retried = bool(d.get("retried", False))
        # Records from before the count was kept say only whether a retry happened.
        retries = int(d.get("retries", 1 if retried else 0))
        return cls(
            name=str(d["name"]),
            noise_floor=float(d["noise_floor"]),
            base_fp=str(d.get("base_fp", "")),
            patched_fp=str(d.get("patched_fp", "")),
            pairs=tuple(PairTiming.from_dict(p) for p in d.get("pairs", [])),
            speedup=None if speedup is None else float(speedup),
            retried=retried,
            errors=tuple(str(e) for e in d.get("errors", [])),
            retries=retries,
        )


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


# Records written before the referee timed more than one input have their single
# input's timing at the top level, with no name for it. They are read into one
# InputTiming under this name. See Measurement.from_dict.
LEGACY_INPUT_NAME = "benchmark"


@dataclass(frozen=True)
class Measurement:
    """Every fact the referee could establish about one patch against the base.

    Steps that could not run leave their field empty or None and add a line to
    ``errors``. Nothing here is a decision. ``clears_noise`` is the one derived
    label: the patch applied, was in scope, passed both suites, computed the same
    result on every input, was timed on every input, was not slower on any of
    them, and was faster on at least one.

    ``speedup`` is the geometric mean across the inputs. A geometric mean
    penalises imbalance by a squared deviation term in log space, so a patch
    that is enormous on one input and flat on the rest scores near the flat
    value: breadth is worth as much as the input count. It does not replace the
    no regression rule, because a mean prices a slowdown as a finite penalty and
    a large enough spike can outrun one. artifacts/generality.md section 7.
    """

    applied: bool = False
    apply_error: str = ""
    scope_violations: tuple[str, ...] = ()
    tests: tuple[SuiteResult, ...] = ()
    canary_s: float | None = None
    inputs: tuple[InputTiming, ...] = ()
    errors: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    wall_s: float = 0.0

    @property
    def tests_pass(self) -> bool:
        scopes = {t.scope: t.ok for t in self.tests}
        return bool(scopes) and all(scopes.values()) and "module" in scopes and "full" in scopes

    @property
    def ratios(self) -> tuple[float, ...]:
        """Every input that produced a ratio. An input whose timing failed has none.

        A ratio is base seconds over patched seconds and both are positive, so
        the zero guard is only so a corrupt record cannot raise out of a
        property.
        """
        return tuple(i.speedup for i in self.inputs if i.speedup is not None and i.speedup > 0)

    @property
    def speedup(self) -> float | None:
        """Geometric mean over the timed inputs, or None if nothing was timed."""
        ratios = self.ratios
        if not ratios:
            return None
        return math.exp(sum(math.log(r) for r in ratios) / len(ratios))

    @property
    def worst_speedup(self) -> float | None:
        ratios = self.ratios
        return min(ratios) if ratios else None

    @property
    def regressions(self) -> tuple[str, ...]:
        """Names of inputs the patch made detectably slower. Empty is the bar."""
        return tuple(i.name for i in self.inputs if i.regresses)

    @property
    def result_matches(self) -> bool | None:
        """True only if every input computed the same thing on both trees."""
        seen = [i.result_matches for i in self.inputs]
        if not seen or any(s is None for s in seen):
            return None
        return all(seen)

    @property
    def untimed(self) -> tuple[str, ...]:
        """Names of inputs that produced no ratio. Empty is the bar.

        The mean is taken over the inputs that did, so a missing input would
        otherwise raise the score exactly when the patch is weakest there. The
        width experiment's two largest recorded speedups were this: three or
        four inputs timed, the sparse ones missing.
        """
        return tuple(i.name for i in self.inputs if i.untimed)

    @property
    def clears_noise(self) -> bool:
        return (
            self.applied
            and not self.scope_violations
            and self.tests_pass
            and self.result_matches is True
            and not self.untimed
            and not self.regressions
            and any(i.improves for i in self.inputs)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "apply_error": self.apply_error,
            "scope_violations": list(self.scope_violations),
            "tests": [t.to_dict() for t in self.tests],
            "tests_pass": self.tests_pass,
            "result_matches": self.result_matches,
            "canary_s": self.canary_s,
            "inputs": [i.to_dict() for i in self.inputs],
            "speedup": self.speedup,
            "worst_speedup": self.worst_speedup,
            "regressions": list(self.regressions),
            "untimed": list(self.untimed),
            "clears_noise": self.clears_noise,
            "errors": list(self.errors),
            "provenance": self.provenance.to_dict(),
            "wall_s": self.wall_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Measurement:
        # Two shapes of older record are read past and never rewritten. An "ir"
        # block is from when the referee ran cachegrind; see
        # artifacts/instruction_counting.md. A top level "pairs" with no
        # "inputs" is from when it timed one benchmark; see
        # artifacts/generality.md. Neither is written again.
        canary = d.get("canary_s")
        return cls(
            applied=bool(d.get("applied", False)),
            apply_error=str(d.get("apply_error", "")),
            scope_violations=tuple(str(v) for v in d.get("scope_violations", [])),
            tests=tuple(SuiteResult.from_dict(t) for t in d.get("tests", [])),
            canary_s=None if canary is None else float(canary),
            inputs=_inputs_from_dict(d),
            errors=tuple(str(e) for e in d.get("errors", [])),
            provenance=Provenance.from_dict(d.get("provenance", {})),
            wall_s=float(d.get("wall_s", 0.0)),
        )


def _inputs_from_dict(d: dict[str, Any]) -> tuple[InputTiming, ...]:
    """The record's inputs, reading a single input record from before the change."""
    if "inputs" in d:
        return tuple(InputTiming.from_dict(i) for i in d["inputs"])
    if "pairs" not in d and "speedup" not in d:
        return ()
    speedup = d.get("speedup")
    return (
        InputTiming(
            name=LEGACY_INPUT_NAME,
            noise_floor=float(d.get("noise_floor", PILOT_NOISE_FLOOR)),
            base_fp=str(d.get("base_fp", "")),
            patched_fp=str(d.get("patched_fp", "")),
            pairs=tuple(PairTiming.from_dict(p) for p in d.get("pairs", [])),
            speedup=None if speedup is None else float(speedup),
        ),
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
