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
    ref: AttemptRef
    incumbent_sha: str
    incumbent_tree: str
    stack_diff: str
    target: TargetSpec
    history: tuple[Attempt, ...]
    docs: tuple[tuple[str, str], ...]
    cache_key: str


class Worker(Protocol):
    def attempt(self, inp: WorkerInput) -> WorkerOutput: ...
