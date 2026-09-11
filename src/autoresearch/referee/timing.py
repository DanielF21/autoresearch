"""The referee's timing arithmetic, all pure.

A pair is one launch of the incumbent tree and one launch of the patched tree,
back to back, so machine drift affects both and cancels. Measurement C showed
this is what keeps the median ratio centred on 1 when nothing changed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from autoresearch.types import PairTiming

INCUMBENT_FIRST = "incumbent_first"
PATCHED_FIRST = "patched_first"


@dataclass(frozen=True)
class PairPlan:
    """What to run for one pair, decided before any timing starts."""

    index: int
    order: str
    hash_seed: int

    @property
    def sequence(self) -> tuple[str, str]:
        if self.order == INCUMBENT_FIRST:
            return ("incumbent", "patched")
        return ("patched", "incumbent")


def plan_pairs(n: int, hash_seeds: tuple[int, ...]) -> tuple[PairPlan, ...]:
    """Alternate which tree goes first so neither always pays a position effect.

    The hash seed rotates through the configured list, one seed per pair, and
    both launches of a pair share it so they see the same dict ordering.
    """
    if n < 1:
        raise ValueError("at least one pair is required")
    if not hash_seeds:
        raise ValueError("at least one hash seed is required")
    return tuple(
        PairPlan(
            index=i,
            order=INCUMBENT_FIRST if i % 2 == 0 else PATCHED_FIRST,
            hash_seed=hash_seeds[i % len(hash_seeds)],
        )
        for i in range(n)
    )


def clean_pairs(pairs: tuple[PairTiming, ...]) -> tuple[PairTiming, ...]:
    return tuple(p for p in pairs if not p.contaminated)


def median_ratio(pairs: tuple[PairTiming, ...]) -> float | None:
    """Median of incumbent over patched across clean pairs. None if no clean pair."""
    clean = clean_pairs(pairs)
    if not clean:
        return None
    return statistics.median(p.ratio for p in clean)


def enough_clean(pairs: tuple[PairTiming, ...], minimum: int) -> bool:
    return len(clean_pairs(pairs)) >= minimum


def is_speedup(median: float | None, threshold: float) -> bool:
    """The acceptance rule: the median ratio must reach the threshold."""
    return median is not None and median >= threshold
