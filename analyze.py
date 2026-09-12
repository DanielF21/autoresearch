#!/usr/bin/env python3
"""Headline numbers and the progress plot for one run.

    uv run analyze.py runs/t1_w4
    uv run analyze.py runs/t1_w4 --out progress.png --csv table.csv

Prints a table of attempt number, wall clock and speedup, and draws
the running best against attempt number: every measured attempt as a grey dot,
every attempt that set a new best in green, and a step line through them.

Wall clock is the median of the patched tree's launches across the referee's
clean pairs, in seconds, so it is the same quantity the speedup is a ratio of.
The baseline is the median of the base tree's launches over the same pairs, and
it is attempt 0 on the plot.

Only an attempt that cleared the noise floor can set a record. That is the same
bar the run itself reports, so a patch that is fast but fails its tests never
appears as progress.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from autoresearch import history
from autoresearch.referee import timing
from autoresearch.types import Attempt, Measurement, PairTiming

LABEL_CHARS = 46


@dataclass(frozen=True)
class Row:
    """One attempt, reduced to what the table and the plot need."""

    number: int
    wall_s: float | None
    speedup: float | None
    clears: bool
    record: bool
    label: str

    @property
    def plotted(self) -> bool:
        return self.wall_s is not None


def _clean(m: Measurement) -> tuple[PairTiming, ...]:
    """The pairs the referee's median is taken over, or all of them if none is clean."""
    return timing.clean_pairs(m.pairs) or m.pairs


def median_patched_s(m: Measurement) -> float | None:
    pairs = _clean(m)
    return statistics.median(p.patched_s for p in pairs) if pairs else None


def median_base_s(m: Measurement) -> float | None:
    pairs = _clean(m)
    return statistics.median(p.base_s for p in pairs) if pairs else None


def short_label(a: Attempt) -> str:
    """The first sentence of the worker's own rationale, cut to fit beside a point."""
    text = " ".join(a.rationale.split())
    if not text:
        return ""
    first = text.split(". ")[0]
    if len(first) <= LABEL_CHARS:
        return first
    cut = first[:LABEL_CHARS]
    return cut[: cut.rindex(" ")] + "…" if " " in cut else cut + "…"


def baseline(attempts: tuple[Attempt, ...]) -> float | None:
    """The unpatched tree's wall clock.

    It is a constant of the run, since every attempt is measured against the
    same commit, so the median across attempts is only a defence against one
    contaminated measurement.
    """
    walls = [
        s
        for a in attempts
        if a.measurement is not None and (s := median_base_s(a.measurement)) is not None
    ]
    return statistics.median(walls) if walls else None


def rows(attempts: tuple[Attempt, ...]) -> list[Row]:
    """One row per attempt, in order, with the running best marked as it moves."""
    best: float | None = None
    out: list[Row] = []
    for a in attempts:
        m = a.measurement
        wall = median_patched_s(m) if m is not None else None
        speedup = m.speedup if m is not None else None
        clears = a.clears_noise
        record = clears and wall is not None and (best is None or wall < best)
        if record:
            best = wall
        out.append(
            Row(
                number=a.ref.number,
                wall_s=wall,
                speedup=speedup,
                clears=clears,
                record=record,
                label=short_label(a),
            )
        )
    return out


def render_table(table: list[Row], base_wall: float | None) -> str:
    def wall(v: float | None) -> str:
        return "--" if v is None else f"{v:.4f}"

    lines = [f"{'attempt':>7}  {'wall_s':>9}  {'speedup':>10}", "-" * 32]
    lines.append(f"{0:>7}  {wall(base_wall):>9}  {'baseline':>10}")
    for r in table:
        mark = " *" if r.record else ""
        speed = "--" if r.speedup is None else f"{r.speedup:.3f}x"
        lines.append(f"{r.number:>7}  {wall(r.wall_s):>9}  {speed:>10}{mark}")
    records = sum(1 for r in table if r.record)
    lines += ["-" * 32, f"{len(table)} attempts, {records} marked * set a new best"]
    return "\n".join(lines)


def write_csv(path: Path, table: list[Row], base_wall: float | None) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attempt", "wall_s", "speedup", "clears_noise", "record"])
        w.writerow([0, base_wall, "", "", ""])
        for r in table:
            w.writerow([r.number, r.wall_s, r.speedup, int(r.clears), int(r.record)])


def _points(rs: list[Row]) -> tuple[list[int], list[float]]:
    """Attempt numbers and seconds for the rows that have a measurement."""
    xs: list[int] = []
    ys: list[float] = []
    for r in rs:
        if r.wall_s is not None:
            xs.append(r.number)
            ys.append(r.wall_s)
    return xs, ys


def plot(
    table: list[Row],
    base_wall: float | None,
    out: Path,
    title: str,
    log: bool = True,
    labels: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    green = "#4bbf73"
    fig, ax = plt.subplots(figsize=(14, 7))

    plotted = [r for r in table if r.plotted]
    others = [r for r in plotted if not r.record]
    records = [r for r in plotted if r.record]

    # The running best as a step: it drops at the attempt that set it and holds
    # until the next one does, which is what makes a flat stretch read as a
    # stretch of no progress rather than as missing data.
    step_x, step_y = _points(records)
    if base_wall is not None:
        step_x.insert(0, 0)
        step_y.insert(0, base_wall)
    if step_x and plotted:
        step_x.append(max(r.number for r in plotted))
        step_y.append(step_y[-1])
    if step_x:
        ax.plot(
            step_x,
            step_y,
            drawstyle="steps-post",
            color=green,
            lw=2,
            zorder=2,
            label="Running best",
        )

    if others:
        grey_x, grey_y = _points(others)
        ax.scatter(grey_x, grey_y, s=18, color="#cccccc", zorder=3, label="No improvement")
    if base_wall is not None:
        ax.scatter([0], [base_wall], s=90, color=green, edgecolor="white", zorder=5)
        ax.annotate(
            "baseline",
            (0, base_wall),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9,
            color="#2f7d4f",
            rotation=20,
        )
    if records:
        best_x, best_y = _points(records)
        ax.scatter(best_x, best_y, s=90, color=green, edgecolor="white", zorder=5, label="New best")
        if labels:
            for x, y, r in zip(best_x, best_y, records, strict=True):
                ax.annotate(
                    r.label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 8),
                    fontsize=8,
                    color="#2f7d4f",
                    rotation=20,
                )

    if log:
        ax.set_yscale("log")
    # The labels lean out to the right of their point, so the axis needs room
    # past the last attempt or the last and most interesting one is clipped.
    if plotted:
        last = max(r.number for r in plotted)
        ax.set_xlim(-0.5, last + (0.22 * last + 1 if labels and records else 0.5))
    ax.set_xlabel("Attempt #")
    ax.set_ylabel("Wall clock seconds (lower is better)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    handles, names = ax.get_legend_handles_labels()
    order = ["No improvement", "New best", "Running best"]
    pairs = sorted(zip(handles, names, strict=True), key=lambda h: order.index(h[1]))
    ax.legend([h for h, _ in pairs], [n for _, n in pairs], loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", help="a run directory, for example runs/t1_w4")
    p.add_argument("--out", default="", help="where the plot goes; default <run>/progress.png")
    p.add_argument("--csv", default="", help="also write the table as csv")
    p.add_argument("--linear", action="store_true", help="linear y axis instead of log")
    p.add_argument("--no-labels", action="store_true", help="no text beside the green points")
    p.add_argument("--no-plot", action="store_true", help="table only")
    args = p.parse_args(argv)

    run = Path(args.run)
    if not (run / "attempts").is_dir():
        print(f"{run} does not look like a run directory", file=sys.stderr)
        return 2
    attempts = history.load_history(history.RunPaths(run))
    if not attempts:
        print(f"no attempts in {run}", file=sys.stderr)
        return 1

    table = rows(attempts)
    base_wall = baseline(attempts)
    print(render_table(table, base_wall))

    if args.csv:
        write_csv(Path(args.csv), table, base_wall)
        print(f"\ntable: {args.csv}")
    if not args.no_plot:
        out = Path(args.out) if args.out else run / "progress.png"
        records = sum(1 for r in table if r.record)
        title = f"Autoresearch Progress: {len(table)} attempts, {records} improvements ({run.name})"
        plot(table, base_wall, out, title, log=not args.linear, labels=not args.no_labels)
        print(f"plot:  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
