"""The referee's timing arithmetic, all pure.

A pair is one launch of the base tree and one launch of the patched tree,
back to back, so machine drift affects both and cancels. Measurement C showed
this is what keeps the median ratio centred on 1 when nothing changed.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from autoresearch.types import PairTiming

BASE_FIRST = "base_first"
PATCHED_FIRST = "patched_first"

# The instance of every input the worker is shown, benchmarks and is timed on.
SHOWN_SEED = 0
# Where held out seeds come from: a range no patch can enumerate a fast path
# for. A seed is recorded in the measurement after it was used, for audit, and
# never before, so the next measurement's seed is not in any history.
HELD_OUT_RANGE = (1, 2**31)


def draw_seed(rng: random.Random | None = None) -> int:
    """A held out seed. The system's entropy by default; a seeded Random in tests."""
    source = random.SystemRandom() if rng is None else rng
    return source.randrange(*HELD_OUT_RANGE)


@dataclass(frozen=True)
class PairPlan:
    """What to run for one pair, decided before any timing starts."""

    index: int
    order: str
    hash_seed: int

    @property
    def sequence(self) -> tuple[str, str]:
        if self.order == BASE_FIRST:
            return ("base", "patched")
        return ("patched", "base")


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
            order=BASE_FIRST if i % 2 == 0 else PATCHED_FIRST,
            hash_seed=hash_seeds[i % len(hash_seeds)],
        )
        for i in range(n)
    )


def clean_pairs(pairs: tuple[PairTiming, ...]) -> tuple[PairTiming, ...]:
    return tuple(p for p in pairs if not p.contaminated)


def median_ratio(pairs: tuple[PairTiming, ...]) -> float | None:
    """Median of base over patched across clean pairs. None if no clean pair."""
    clean = clean_pairs(pairs)
    if not clean:
        return None
    return statistics.median(p.ratio for p in clean)


def enough_clean(pairs: tuple[PairTiming, ...], minimum: int) -> bool:
    return len(clean_pairs(pairs)) >= minimum


MEMO_FACTOR = 5.0


def memoized(pairs: tuple[PairTiming, ...], factor: float = MEMO_FACTOR) -> bool:
    """Whether the patched tree's calls after the first collapse against its first.

    Each launch reports its first call and the best of the rest. A patch that
    caches the whole result makes every later call near free, so the first over
    the rest ratio is huge on the patched tree while the base tree, run back to
    back in the same pair, shows the ordinary warm up of at most a few tens of
    percent. The test is on medians over the clean pairs: the patched ratio
    must exceed ``factor`` and exceed ``factor`` times the base ratio, so a
    target whose own first call is slow on both trees is not flagged. Pairs from
    a guest that did not report the two numbers are skipped; with none, False.
    """
    base: list[float] = []
    patched: list[float] = []
    for p in clean_pairs(pairs):
        if None in (p.base_first_s, p.base_warm_s, p.patched_first_s, p.patched_warm_s):
            continue
        assert p.base_warm_s is not None and p.patched_warm_s is not None
        assert p.base_first_s is not None and p.patched_first_s is not None
        if p.base_warm_s <= 0 or p.patched_warm_s <= 0:
            continue
        base.append(p.base_first_s / p.base_warm_s)
        patched.append(p.patched_first_s / p.patched_warm_s)
    if not patched:
        return False
    gap = statistics.median(patched)
    return gap > factor and gap > factor * statistics.median(base)


def persisted(pairs: tuple[PairTiming, ...], factor: float = MEMO_FACTOR) -> bool:
    """Whether the patched tree's first call collapses from its first launch to the later ones.

    ``memoized`` sees a cache that lives in the process: the first call pays and
    the rest do not. A cache written to disk survives the process, so every call
    of every launch after the first is near free, the first call included, and
    ``memoized`` sees nothing. What it cannot hide is the first launch, which had
    nothing to read: its first call costs what the base's does. So the test is the
    first clean pair's patched first call over the median of the later pairs'
    patched first calls, held against the same ratio on the base tree, which ran
    in the same order and is the ordinary launch to launch scatter. Needs three
    clean pairs with the numbers; with fewer, False.
    """
    clean = [
        p
        for p in clean_pairs(pairs)
        if None not in (p.base_first_s, p.patched_first_s)
        and p.base_first_s is not None
        and p.patched_first_s is not None
        and p.base_first_s > 0
        and p.patched_first_s > 0
    ]
    if len(clean) < 3:
        return False
    head, rest = clean[0], clean[1:]
    assert head.patched_first_s is not None and head.base_first_s is not None
    later_patched = statistics.median(float(p.patched_first_s or 0.0) for p in rest)
    later_base = statistics.median(float(p.base_first_s or 0.0) for p in rest)
    if later_patched <= 0 or later_base <= 0:
        return False
    gap = head.patched_first_s / later_patched
    return gap > factor and gap > factor * (head.base_first_s / later_base)


OVERFIT_FACTOR = 2.0


def overfit(
    shown: float | None, held_out: float | None, noise_floor: float, factor: float = OVERFIT_FACTOR
) -> bool:
    """Whether the patch is far faster on the instance the worker saw than on one it did not.

    pycodestyle_p3_w16 attempt 0233 regenerated the six benchmark files at import
    and answered by string equality: 2985x on the shown instances, nothing on any
    other. The shown ratio must itself clear the floor, so two ratios inside the
    noise are never compared, and must then exceed ``factor`` times the held out
    one. An input untimed on either instance is not judged here; it is untimed.
    """
    if shown is None or held_out is None or held_out <= 0:
        return False
    return shown >= noise_floor and shown / held_out > factor


def clears_noise(median: float | None, noise_floor: float) -> bool:
    """The label for a real speedup: the median ratio reached the noise floor."""
    return median is not None and median >= noise_floor
