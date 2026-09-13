"""A noise floor per input, from pairs of two unpatched trees of the base commit.

Nothing is patched, so every ratio is what a change of nothing looks like. The
floor for an input is the ratio a median of ``[referee].pairs`` such ratios
exceeds only once in ``1 / FALSE_ALARM``, bootstrapped from the ratios measured.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from autoresearch.referee import timing
from autoresearch.referee.thresholds import null_threshold
from autoresearch.types import PairTiming

FALSE_ALARM = 0.01


def floor_for(ratios: tuple[float, ...], pairs: int, false_alarm: float) -> float:
    """The floor rounded up to four decimals, so the config never sits under it."""
    raw = null_threshold(ratios, pairs=pairs, false_alarm=false_alarm)
    return -(-raw * 10_000 // 1) / 10_000


def report(
    name: str, all_pairs: tuple[PairTiming, ...], pairs_per_measure: int, false_alarm: float
) -> tuple[str, dict[str, Any]]:
    clean = timing.clean_pairs(all_pairs)
    ratios = tuple(p.ratio for p in clean)
    if len(ratios) < 2:
        return (
            f"{name:<14}  only {len(ratios)} clean of {len(all_pairs)} pairs, no floor",
            {"input": name, "clean": len(ratios), "total": len(all_pairs), "floor": None},
        )
    floor = floor_for(ratios, pairs_per_measure, false_alarm)
    lo, hi = min(ratios), max(ratios)
    line = (
        f"{name:<14}  floor {floor:.4f}   from {len(ratios)} clean of {len(all_pairs)} pairs, "
        f"median {statistics.median(ratios):.4f}, range {lo:.4f} to {hi:.4f}"
    )
    return line, {
        "input": name,
        "clean": len(ratios),
        "total": len(all_pairs),
        "median": statistics.median(ratios),
        "min": lo,
        "max": hi,
        "floor": floor,
        "pairs_per_measure": pairs_per_measure,
        "false_alarm": false_alarm,
    }


def write_record(
    path: Path,
    header: dict[str, Any],
    records: list[dict[str, Any]],
    measured: dict[str, tuple[PairTiming, ...]],
) -> None:
    """Every floor and every pair, so a floor can be recomputed without another box."""
    with path.open("w") as fh:
        fh.write(json.dumps({"kind": "run", **header}) + "\n")
        for rec in records:
            fh.write(json.dumps({"kind": "floor", **rec}) + "\n")
        for name, pairs in measured.items():
            for pair in pairs:
                fh.write(json.dumps({"kind": "pair", "input": name, **pair.to_dict()}) + "\n")
