"""Which mergeable candidates are worth choosing between.

A candidate is dominated when another is at least as fast and at least as small,
and strictly better on one of the two. Speed is the measured geometric mean and
size is lines added plus removed. What remains is the frontier. One point wins
outright; several go to the comparison call, because code cannot say whether 3%
more speed is worth 33 more lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Point:
    number: int
    speedup: float
    lines: int


def pareto_frontier(points: list[Point]) -> tuple[Point, ...]:
    def dominated(p: Point) -> bool:
        return any(
            q.speedup >= p.speedup
            and q.lines <= p.lines
            and (q.speedup > p.speedup or q.lines < p.lines)
            for q in points
        )

    return tuple(sorted((p for p in points if not dominated(p)), key=lambda p: -p.speedup))


@dataclass(frozen=True)
class Selection:
    considered: tuple[Point, ...]
    frontier: tuple[Point, ...]
    excluded: dict[int, str] = field(default_factory=dict)
    pick: int | None = None
    rule: str = ""
    reasons: str = ""

    def to_dict(self) -> dict[str, Any]:
        def pts(ps: tuple[Point, ...]) -> list[dict[str, Any]]:
            return [{"number": p.number, "speedup": p.speedup, "lines": p.lines} for p in ps]

        return {
            "considered": pts(self.considered),
            "frontier": pts(self.frontier),
            "excluded": {str(k): v for k, v in sorted(self.excluded.items())},
            "pick": self.pick,
            "rule": self.rule,
            "reasons": self.reasons,
        }
