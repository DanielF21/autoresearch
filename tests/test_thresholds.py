"""The noise floor in configs/t1_w1.toml is derived, not typed. These tests keep it so."""

import json
from pathlib import Path

import pytest

from autoresearch.config import load_config
from autoresearch.referee.thresholds import (
    PILOT_FALSE_ALARM,
    PILOT_NOISE_FLOOR,
    PILOT_PAIRS,
    null_ratios_from_measure_c,
    null_threshold,
)

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "measure_c_null_ratios_long.json"
RAW = [
    ROOT / "runs" / "measure-cd" / "20260911-124254-c0.jsonl",
    ROOT / "runs" / "measure-cd" / "20260911-124254-c1.jsonl",
]


def _fixture_ratios() -> tuple[float, ...]:
    return tuple(float(x) for x in json.loads(FIXTURE.read_text())["ratios"])


def test_fixture_has_48_null_pairs() -> None:
    assert len(_fixture_ratios()) == 48


@pytest.mark.skipif(not all(p.exists() for p in RAW), reason="raw measurement C data not present")
def test_fixture_matches_raw_measurement_c() -> None:
    assert null_ratios_from_measure_c(RAW, "long") == _fixture_ratios()


def test_pilot_noise_floor_is_reproduced_from_the_null_data() -> None:
    computed = null_threshold(_fixture_ratios(), PILOT_PAIRS, PILOT_FALSE_ALARM)
    assert computed == pytest.approx(PILOT_NOISE_FLOOR, abs=0.0003)


def test_config_noise_floor_equals_the_derived_constant() -> None:
    cfg = load_config(ROOT / "configs" / "t1_w1.toml")
    assert cfg.referee.pairs == PILOT_PAIRS
    assert cfg.referee.noise_floor == PILOT_NOISE_FLOOR


@pytest.mark.parametrize(
    ("pairs", "expected_pct"),
    [(1, 1.65), (3, 1.12), (5, 0.72), (10, 0.56), (24, 0.41)],
)
def test_reproduces_the_measurement_c_resolution_table(pairs: int, expected_pct: float) -> None:
    """artifacts/measurements.md, resolution curve, long graph, 5 percent false alarm."""
    t = null_threshold(_fixture_ratios(), pairs, 0.05)
    assert 100 * (t - 1) == pytest.approx(expected_pct, abs=0.02)


def test_more_pairs_lower_the_floor() -> None:
    ratios = _fixture_ratios()
    t = [null_threshold(ratios, k, 0.05, resamples=5000) for k in (1, 3, 6, 12)]
    assert t == sorted(t, reverse=True)


def test_null_threshold_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        null_threshold((1.0,), 3, 0.05)
    with pytest.raises(ValueError):
        null_threshold((1.0, 1.1), 3, 1.5)
