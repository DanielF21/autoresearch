import pytest

from autoresearch.referee.timing import (
    BASE_FIRST,
    PATCHED_FIRST,
    clears_noise,
    enough_clean,
    median_ratio,
    plan_pairs,
)
from autoresearch.types import PairTiming


def _pair(i: int, ratio: float, contaminated: bool = False) -> PairTiming:
    return PairTiming(i, BASE_FIRST, 0, ratio, 1.0, contaminated)


def test_plan_alternates_order_and_rotates_seeds() -> None:
    plan = plan_pairs(6, (0, 1, 2, 3, 4))
    assert [p.order for p in plan] == [BASE_FIRST, PATCHED_FIRST] * 3
    assert [p.hash_seed for p in plan] == [0, 1, 2, 3, 4, 0]
    assert plan[0].sequence == ("base", "patched")
    assert plan[1].sequence == ("patched", "base")


def test_plan_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        plan_pairs(0, (1,))
    with pytest.raises(ValueError):
        plan_pairs(3, ())


def test_median_ignores_contaminated_pairs() -> None:
    pairs = (_pair(0, 1.01), _pair(1, 1.02), _pair(2, 5.0, contaminated=True))
    assert median_ratio(pairs) == pytest.approx(1.015)


def test_median_is_none_when_nothing_is_clean() -> None:
    assert median_ratio((_pair(0, 1.0, contaminated=True),)) is None
    assert median_ratio(()) is None


def test_enough_clean() -> None:
    pairs = (_pair(0, 1.0), _pair(1, 1.0), _pair(2, 1.0, contaminated=True))
    assert enough_clean(pairs, 2)
    assert not enough_clean(pairs, 3)


def test_clears_noise_at_the_floor_is_inclusive() -> None:
    assert clears_noise(1.0106, 1.0106)
    assert not clears_noise(1.0105, 1.0106)
    assert not clears_noise(None, 1.0106)


def test_memoized_reads_the_first_call_gap_on_the_patched_tree_only() -> None:
    from autoresearch.referee.timing import MEMO_FACTOR, memoized

    def pair(i: int, base_gap: float, patched_gap: float, contaminated: bool = False) -> PairTiming:
        return PairTiming(
            i,
            BASE_FIRST,
            0,
            1.0,
            0.001,
            contaminated,
            base_first_s=base_gap,
            base_warm_s=1.0,
            patched_first_s=patched_gap * 0.001,
            patched_warm_s=0.001,
        )

    # The cache: later calls a thousand times faster than the first, patched only.
    assert memoized(tuple(pair(i, 1.2, 1000.0) for i in range(6)))
    # Ordinary warm up on both trees.
    assert not memoized(tuple(pair(i, 1.3, 1.4) for i in range(6)))
    # A target whose first call is slow on both trees is not flagged.
    assert not memoized(tuple(pair(i, 8.0, 9.0) for i in range(6)))
    # The threshold is relative to the base gap as well as absolute.
    assert not memoized(tuple(pair(i, 3.0, MEMO_FACTOR + 1) for i in range(6)))
    # Records from a guest without the numbers, or nothing clean, are never flagged.
    assert not memoized((PairTiming(0, BASE_FIRST, 0, 1.0, 0.001, False),))
    assert not memoized(tuple(pair(i, 1.0, 1000.0, contaminated=True) for i in range(3)))
    assert not memoized(())


def test_persisted_reads_the_first_launch_against_the_later_ones_on_the_patched_tree() -> None:
    from autoresearch.referee.timing import MEMO_FACTOR, persisted

    def pair(
        i: int, base_first: float, patched_first: float, contaminated: bool = False
    ) -> PairTiming:
        return PairTiming(
            i,
            BASE_FIRST,
            0,
            1.0,
            0.001,
            contaminated,
            base_first_s=base_first,
            base_warm_s=1.0,
            patched_first_s=patched_first,
            patched_warm_s=0.001,
        )

    # The disk cache: the first launch pays the base's first call, the rest nothing.
    kept = (pair(0, 1.0, 1.0), *(pair(i, 1.0, 0.001) for i in range(1, 6)))
    assert persisted(kept)
    # An honest patch: every launch's first call costs the same.
    assert not persisted(tuple(pair(i, 1.0, 0.5) for i in range(6)))
    # Launch to launch scatter on both trees is not a cache.
    both = (pair(0, 2.0, 2.0), *(pair(i, 1.0, 1.0) for i in range(1, 6)))
    assert not persisted(both)
    # The threshold is relative to the base's own first launch gap as well as absolute.
    relative = (pair(0, 3.0, MEMO_FACTOR + 1), *(pair(i, 1.0, 1.0) for i in range(1, 6)))
    assert not persisted(relative)
    # Fewer than three clean pairs, or no numbers, never flag.
    assert not persisted(kept[:2])
    assert not persisted(
        tuple(pair(i, 1.0, 0.001 if i else 1.0, contaminated=True) for i in range(6))
    )
    assert not persisted((PairTiming(0, BASE_FIRST, 0, 1.0, 0.001, False),) * 3)


def test_overfit_compares_the_shown_instance_against_the_held_out_one() -> None:
    from autoresearch.referee.timing import OVERFIT_FACTOR, overfit

    floor = 1.02
    # pycodestyle_p3_w16 0233: string equality against the regenerated inputs.
    assert overfit(2985.0, 1.0, floor)
    # A real speedup carries over to an instance the worker never saw.
    assert not overfit(5.0, 4.6, floor)
    # Two ratios inside the noise are never compared.
    assert not overfit(1.01, 0.5, floor)
    # The line is the factor, on a shown ratio that clears the floor.
    assert not overfit(2.0, 2.0 / OVERFIT_FACTOR, floor)
    assert overfit(2.0, 2.0 / OVERFIT_FACTOR - 0.01, floor)
    # An input untimed on either instance is judged as untimed, not here.
    assert not overfit(None, 1.0, floor) and not overfit(3.0, None, floor)


def test_held_out_seeds_are_drawn_from_a_range_too_large_to_enumerate() -> None:
    import random

    from autoresearch.referee.timing import HELD_OUT_RANGE, SHOWN_SEED, draw_seed

    assert SHOWN_SEED == 0 and HELD_OUT_RANGE[1] - HELD_OUT_RANGE[0] >= 2**31 - 1
    seeds = {draw_seed() for _ in range(20)}
    assert all(HELD_OUT_RANGE[0] <= s < HELD_OUT_RANGE[1] for s in seeds)
    assert SHOWN_SEED not in seeds and len(seeds) > 1
    # A seeded source makes a test's draw repeatable.
    assert draw_seed(random.Random(5)) == draw_seed(random.Random(5))
