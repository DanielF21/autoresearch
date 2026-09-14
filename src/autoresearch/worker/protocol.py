"""The worker interface: one attempt in, one output out.

The orchestrator builds a WorkerInput from the run directory and calls
``attempt``. The worker owns nothing between calls. That is what lets the
agent loop be swapped for something else without the orchestrator noticing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from autoresearch.config import TargetSpec
from autoresearch.types import Attempt, AttemptRef, WorkerOutput


@dataclass(frozen=True)
class WorkerInput:
    """``base_sha`` is the run's base commit. It never changes; every attempt
    starts from it and every earlier attempt in ``history`` was measured against it."""

    ref: AttemptRef
    base_sha: str
    target: TargetSpec
    history: tuple[Attempt, ...]
    docs: tuple[tuple[str, str], ...]
    cache_key: str
    prompt: str = "v1"  # the system prompt version this slot runs with
    # Attempt numbers the orchestrator withheld from this slot: absent from
    # ``history`` and so from the box's history directory. Empty for a slot that
    # sees everything, which is every slot outside the hidden leader experiment.
    hidden_numbers: tuple[int, ...] = ()


class Worker(Protocol):
    def attempt(self, inp: WorkerInput) -> WorkerOutput: ...
