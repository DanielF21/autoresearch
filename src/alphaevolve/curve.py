"""Cumulative tokens against the best improvement so far, from any run directory.

Both arms write the harness's records, so one reader serves both. One point per
completed round: the round's cumulative tokens from ``rounds.jsonl``, the number
of referee measurements so far, and the best held out speedup among attempts
that clear the noise floor, recomputed from ``measurement.json`` by today's
``Measurement`` so every run is judged by the same rules. Round zero is the
original code at zero tokens.

A new best reaches the next batch only when its round ends, so a round is the
honest resolution. Curves are cut at one shared budget: a point counts only if
its cumulative tokens are within it, so a batch that overshoots is not compared.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from alphaevolve.budget import primary, uncached
from autoresearch import history
from autoresearch.types import Usage


@dataclass(frozen=True)
class Point:
    round: int
    tokens: int
    uncached: int
    measurements: int
    best: float | None


def points(run_dir: Path) -> tuple[Point, ...]:
    paths = history.RunPaths(run_dir)
    attempts = history.load_history(paths)
    out = [Point(0, 0, 0, 0, None)]
    usage = Usage()
    for record in history.read_rounds(paths):
        usage = usage + record.usage
        upto = tuple(a for a in attempts if a.ref.round <= record.round)
        best, _ = history.best_ratio(upto)
        measured = sum(1 for a in upto if a.measurement is not None)
        out.append(Point(record.round, primary(usage), uncached(usage), measured, best))
    return tuple(out)


def cut(curve: Sequence[Point], budget: int) -> tuple[Point, ...]:
    return tuple(p for p in curve if p.tokens <= budget)


def write_csv(curves: Mapping[str, Sequence[Point]], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "round", "tokens", "uncached_tokens", "measurements", "best"])
        for label, curve in curves.items():
            for p in curve:
                best = "" if p.best is None else f"{p.best:.6f}"
                writer.writerow([label, p.round, p.tokens, p.uncached, p.measurements, best])


def plot(curves: Mapping[str, Sequence[Point]], path: Path, budget: int) -> None:
    """A step plot on a log token axis. matplotlib is a development dependency,
    so it is imported here and nowhere a run imports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curve in curves.items():
        xs = [max(p.tokens, 1) for p in curve]
        ys = [1.0 if p.best is None else p.best for p in curve]
        ax.step(xs, ys, where="post", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("tokens spent, prompt plus completion")
    ax.set_ylabel("best held out speedup so far (x)")
    ax.axvline(budget, linestyle=":", color="grey", label=f"budget {budget:,}")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
