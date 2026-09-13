"""The run directory: the only state in the system, and how it is read and written.

Layout, relative to the run directory:

    config.toml           frozen copy of the run config
    target/               files the worker is shown about the target
    attempts/NNNN/        one directory per attempt, written once
    rounds.jsonl          one record per completed round
    boxes.json            referee box ids while held, and any that failed to terminate;
                          the next launch terminates whatever it names
    LOCK                  owner of the running orchestrator

An attempt directory is written by ``write_attempt`` and never modified, except
that ``write_measurement`` adds ``measurement.json`` once the referee has run.
``load_history`` is the one read path the worker's view goes through.

There is no incumbent. Every attempt starts from, and is measured against, the
run's base commit, which is the target's pinned sha and never changes.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.types import (
    Attempt,
    AttemptRef,
    Measurement,
    Prediction,
    RoundRecord,
    StopReason,
    Usage,
    WorkerOutput,
)

CONFIG_FILE = "config.toml"
TARGET_DIR = "target"
ATTEMPTS_DIR = "attempts"
ROUNDS_FILE = "rounds.jsonl"
BOXES_FILE = "boxes.json"
LOCK_FILE = "LOCK"

INPUT_JSON = "input.json"
OUTPUT_JSON = "output.json"
PATCH_DIFF = "patch.diff"
PREDICTION_JSON = "prediction.json"
RATIONALE_MD = "rationale.md"
MEASUREMENT_JSON = "measurement.json"
TRANSCRIPT_JSONL = "transcript.jsonl"
USAGE_JSON = "usage.json"


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def target(self) -> Path:
        return self.root / TARGET_DIR

    @property
    def attempts(self) -> Path:
        return self.root / ATTEMPTS_DIR

    @property
    def rounds(self) -> Path:
        return self.root / ROUNDS_FILE

    @property
    def boxes(self) -> Path:
        return self.root / BOXES_FILE

    @property
    def lock(self) -> Path:
        return self.root / LOCK_FILE

    def attempt(self, ref: AttemptRef) -> Path:
        return self.attempts / ref.dirname


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_attempt(
    paths: RunPaths,
    ref: AttemptRef,
    base_sha: str,
    input_extra: dict[str, Any],
    output: WorkerOutput,
    transcript: str,
    *,
    skipped: str = "",
    duplicate_of: str = "",
) -> Path:
    """Create the attempt directory. Refuses to overwrite: attempts are write once.

    ``skipped`` records why the referee will not run, which is only ever that
    there was no patch. ``duplicate_of`` names an earlier attempt with the same
    normalised diff; the referee runs anyway.
    """
    d = paths.attempt(ref)
    if d.exists():
        raise HistoryError(f"attempt directory already exists: {d}")
    d.mkdir(parents=True)
    _write_json(
        d / INPUT_JSON,
        {
            "ref": ref.to_dict(),
            "base_sha": base_sha,
            "skipped": skipped,
            "duplicate_of": duplicate_of,
            **input_extra,
        },
    )
    _write_json(d / OUTPUT_JSON, output.to_dict())
    _write_json(d / USAGE_JSON, output.usage.to_dict())
    if output.patch is not None:
        (d / PATCH_DIFF).write_text(output.patch)
    if output.prediction is not None:
        _write_json(d / PREDICTION_JSON, output.prediction.to_dict())
    (d / RATIONALE_MD).write_text(output.rationale)
    (d / TRANSCRIPT_JSONL).write_text(transcript)
    return d


def write_measurement(paths: RunPaths, ref: AttemptRef, measurement: Measurement) -> None:
    d = paths.attempt(ref)
    target = d / MEASUREMENT_JSON
    if target.exists():
        raise HistoryError(f"measurement already written: {target}")
    _write_json(target, measurement.to_dict())


def read_attempt(d: Path) -> Attempt:
    """Read one attempt directory. The measurement is included when present."""
    inp = _read_json(d / INPUT_JSON)
    out = _read_json(d / OUTPUT_JSON)
    patch_path = d / PATCH_DIFF
    pred_path = d / PREDICTION_JSON
    measurement_path = d / MEASUREMENT_JSON
    return Attempt(
        ref=AttemptRef.from_dict(inp["ref"]),
        base_sha=str(inp["base_sha"]),
        patch=patch_path.read_text() if patch_path.exists() else None,
        prediction=Prediction.from_dict(_read_json(pred_path)) if pred_path.exists() else None,
        rationale=(d / RATIONALE_MD).read_text() if (d / RATIONALE_MD).exists() else "",
        stop_reason=StopReason(out["stop_reason"]),
        usage=Usage.from_dict(out.get("usage", {})),
        wall_s=float(out.get("wall_s", 0.0)),
        measurement=(
            Measurement.from_dict(_read_json(measurement_path))
            if measurement_path.exists()
            else None
        ),
        skipped=str(inp.get("skipped", "")),
        duplicate_of=str(inp.get("duplicate_of", "")),
    )


def load_history(paths: RunPaths) -> tuple[Attempt, ...]:
    """Every attempt on disk, in attempt number order. The worker's whole view of the past."""
    if not paths.attempts.exists():
        return ()
    dirs = sorted(p for p in paths.attempts.iterdir() if p.is_dir() and (p / INPUT_JSON).exists())
    return tuple(read_attempt(d) for d in dirs)


def next_attempt_number(paths: RunPaths) -> int:
    history = load_history(paths)
    return 1 if not history else history[-1].ref.number + 1


def best_ratio(history: tuple[Attempt, ...]) -> tuple[float | None, int | None]:
    """The best speedup among attempts that clear the noise floor, and which attempt.

    The speedup is the geometric mean over the target's inputs, and clearing the
    floor already requires that no input got slower, so this cannot select a
    patch that traded one input for another.
    """
    best: float | None = None
    which: int | None = None
    for a in history:
        if a.measurement is None or not a.measurement.clears_noise:
            continue
        ratio = a.measurement.speedup
        if ratio is not None and (best is None or ratio > best):
            best, which = ratio, a.ref.number
    return best, which


def append_round(paths: RunPaths, record: RoundRecord) -> None:
    with paths.rounds.open("a") as fh:
        fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_rounds(paths: RunPaths) -> tuple[RoundRecord, ...]:
    if not paths.rounds.exists():
        return ()
    out = []
    for line in paths.rounds.read_text().splitlines():
        if line.strip():
            out.append(RoundRecord.from_dict(json.loads(line)))
    return tuple(out)


def last_completed_round(paths: RunPaths) -> int:
    rounds = read_rounds(paths)
    return 0 if not rounds else rounds[-1].round


def write_boxes(paths: RunPaths, boxes: dict[str, str]) -> None:
    _write_json(paths.boxes, boxes)


def read_boxes(paths: RunPaths) -> dict[str, str]:
    if not paths.boxes.exists():
        return {}
    data = _read_json(paths.boxes)
    return {str(k): str(v) for k, v in data.items()}


def acquire_lock(paths: RunPaths) -> None:
    """Refuse to run two orchestrators on one run directory.

    The lock records host and pid. It is only advisory: a crashed orchestrator
    leaves it behind, and ``release_lock`` or a manual delete clears it.
    """
    if paths.lock.exists():
        raise HistoryError(
            f"run is locked by {paths.lock.read_text().strip()}; delete LOCK if stale"
        )
    paths.lock.write_text(f"{socket.gethostname()}:{os.getpid()}\n")


def release_lock(paths: RunPaths) -> None:
    paths.lock.unlink(missing_ok=True)
