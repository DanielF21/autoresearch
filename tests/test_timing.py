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
