"""Where the acceptance threshold comes from.

Measurement C timed two copies of networkx that differed by a blank line, 48
pairs on the large graph. Those 48 ratios are what a patch that changes nothing
looks like. The threshold for k pairs at false alarm rate a is the (1 - a)
quantile of the median of k ratios drawn from that null set. A patch whose
median clears it has less than a chance of being noise.

``PILOT_THRESHOLD`` is the value for 6 pairs at 1 in 100, and
``tests/test_thresholds.py`` recomputes it from the raw JSONL so the number in
the config is checked rather than typed.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Iterable
from pathlib import Path

PILOT_PAIRS = 6
PILOT_FALSE_ALARM = 0.01
PILOT_THRESHOLD = 1.0106


def null_threshold(
    null_ratios: tuple[float, ...],
    pairs: int,
    false_alarm: float,
    resamples: int = 20000,
    seed: int = 1,
) -> float:
    """The ratio a median of ``pairs`` null draws exceeds with probability ``false_alarm``."""
    if len(null_ratios) < 2:
        raise ValueError("need at least two null ratios")
    if not 0 < false_alarm < 1:
        raise ValueError("false_alarm must be between 0 and 1")
    rng = random.Random(seed)
    n = len(null_ratios)
    medians = sorted(
        statistics.median(null_ratios[rng.randrange(n)] for _ in range(pairs))
        for _ in range(resamples)
    )
    index = min(resamples - 1, int((1.0 - false_alarm) * resamples))
    return medians[index]


def null_ratios_from_measure_c(paths: Iterable[Path], bench: str = "long") -> tuple[float, ...]:
    """Recover the null pair ratios from measurement C's JSONL files.

    Each quartet has four launches ordered by ``pos``: ABBA or BAAB. Positions 0 and 1
    form one pair, 2 and 3 the other. The ratio is tree A's best time over tree B's,
    which under the null is just one tree against another.
    """
    ratios: list[float] = []
    for path in paths:
        launches = [
            r
            for r in _records(path)
            if r.get("kind") == "launch" and not r.get("error") and r.get("bench") == bench
        ]
        for q in sorted({int(str(r["q"])) for r in launches}):
            quartet = sorted(
                (r for r in launches if int(str(r["q"])) == q), key=lambda r: int(str(r["pos"]))
            )
            if len(quartet) != 4:
                continue
            for i, j in ((0, 1), (2, 3)):
                x, y = quartet[i], quartet[j]
                a, b = (x, y) if x["arm"] == "A" else (y, x)
                ratios.append(_best(a) / _best(b))
    return tuple(ratios)


def _best(rec: dict[str, object]) -> float:
    clean = rec.get("min_clean")
    value = clean if clean is not None else rec.get("min_all")
    if not isinstance(value, int | float):
        raise ValueError("launch record has no timing")
    return float(value)


def _records(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            out.append(json.loads(line))
    return out
