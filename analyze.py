#!/usr/bin/env python3
"""Headline numbers and the progress plot for one run.

    uv run analyze.py runs/t1_w4c
    uv run analyze.py runs/t1_w4c --out progress.png --csv table.csv

Prints a table of attempt number, speedup and every input's own ratio, and draws
the running best speedup against attempt number: every measured attempt as a
grey dot, every attempt that set a new best in green, and a step line through
them.

Speedup is the geometric mean over the target's inputs, the number the referee
records. The baseline is 1.0x by definition, drawn as a flat line, since a
speedup is already relative to the unpatched tree.

Only an attempt that cleared the noise floor can set a record, which requires
that no input got slower. That is the same bar the run itself reports, so a
patch that is fast on one input but slower on another never appears as progress
however large its mean.

A run recorded before the referee timed more than one input loads with a single
input named "benchmark", so old runs still plot.
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
from autoresearch.types import Attempt, InputTiming, PairTiming

LABEL_CHARS = 46
BASELINE = 1.0


@dataclass(frozen=True)
class Row:
    """One attempt, reduced to what the table and the plot need."""

    number: int
    speedup: float | None
    worst: float | None
    per_input: dict[str, float | None]
    regressions: tuple[str, ...]
    clears: bool
    record: bool
    label: str

    @property
    def plotted(self) -> bool:
        return self.speedup is not None


def _clean(t: InputTiming) -> tuple[PairTiming, ...]:
    """The pairs the referee's median is taken over, or all of them if none is clean."""
    return timing.clean_pairs(t.pairs) or t.pairs


def median_patched_s(t: InputTiming) -> float | None:
    pairs = _clean(t)
    return statistics.median(p.patched_s for p in pairs) if pairs else None


def median_base_s(t: InputTiming) -> float | None:
    pairs = _clean(t)
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


def input_names(attempts: tuple[Attempt, ...]) -> list[str]:
    """Every input name seen, in the order the referee recorded them.

    Read from the measurements rather than a config so an old run, or one whose
    inputs changed, still tabulates.
    """
    names: list[str] = []
    for a in attempts:
        if a.measurement is None:
            continue
        for i in a.measurement.inputs:
            if i.name not in names:
                names.append(i.name)
    return names


def rows(attempts: tuple[Attempt, ...]) -> list[Row]:
    """One row per attempt, in order, with the running best marked as it moves."""
    best: float | None = None
    out: list[Row] = []
    for a in attempts:
        m = a.measurement
        speedup = m.speedup if m is not None else None
        clears = a.clears_noise
        record = clears and speedup is not None and (best is None or speedup > best)
        if record:
            best = speedup
        out.append(
            Row(
                number=a.ref.number,
                speedup=speedup,
                worst=m.worst_speedup if m is not None else None,
                per_input={i.name: i.speedup for i in m.inputs} if m is not None else {},
                regressions=m.regressions if m is not None else (),
                clears=clears,
                record=record,
                label=short_label(a),
            )
        )
    return out


def _ratio(v: float | None) -> str:
    return "--" if v is None else f"{v:.3f}x"


def render_table(table: list[Row], names: list[str]) -> str:
    # Columns are as wide as the longest input name, so no name is cut.
    col = max([11, *(len(n) for n in names)])
    head = f"{'attempt':>7}  {'geomean':>10}  {'worst':>9}" + "".join(
        f"  {n:>{col}}" for n in names
    )
    width = len(head) + 2
    lines = [head, "-" * width]
    lines.append(
        f"{0:>7}  {'baseline':>10}  {'1.000x':>9}" + "".join(f"  {'1.000x':>{col}}" for _ in names)
    )
    for r in table:
        mark = " *" if r.record else ""
        cells = "".join(f"  {_ratio(r.per_input.get(n)):>{col}}" for n in names)
        lines.append(f"{r.number:>7}  {_ratio(r.speedup):>10}  {_ratio(r.worst):>9}{cells}{mark}")
    records = sum(1 for r in table if r.record)
    slower = sum(1 for r in table if r.regressions)
    lines += [
        "-" * width,
        f"{len(table)} attempts, {records} marked * set a new best, "
        f"{slower} were slower on at least one input",
    ]
    return "\n".join(lines)


def write_csv(path: Path, table: list[Row], names: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["attempt", "speedup", "worst", *names, "regressions", "clears_noise", "record"])
        w.writerow([0, BASELINE, BASELINE, *[BASELINE for _ in names], "", "", ""])
        for r in table:
            w.writerow(
                [
                    r.number,
                    r.speedup,
                    r.worst,
                    *[r.per_input.get(n) for n in names],
                    " ".join(r.regressions),
                    int(r.clears),
                    int(r.record),
                ]
            )


def _points(rs: list[Row]) -> tuple[list[int], list[float]]:
    """Attempt numbers and speedups for the rows that have a measurement."""
    xs: list[int] = []
    ys: list[float] = []
    for r in rs:
        if r.speedup is not None:
            xs.append(r.number)
            ys.append(r.speedup)
    return xs, ys


def plot(
    table: list[Row],
    out: Path,
    title: str,
    log: bool = True,
    labels: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    green = "#4bbf73"
    red = "#d16565"
    fig, ax = plt.subplots(figsize=(14, 7))

    plotted = [r for r in table if r.plotted]
    records = [r for r in plotted if r.record]
    slower = [r for r in plotted if not r.record and r.regressions]
    others = [r for r in plotted if not r.record and not r.regressions]

    # The running best as a step: it rises at the attempt that set it and holds
    # until the next one does, which is what makes a flat stretch read as a
    # stretch of no progress rather than as missing data.
    step_x, step_y = _points(records)
    step_x.insert(0, 0)
    step_y.insert(0, BASELINE)
    if plotted:
        step_x.append(max(r.number for r in plotted))
        step_y.append(step_y[-1])
    ax.plot(
        step_x, step_y, drawstyle="steps-post", color=green, lw=2, zorder=2, label="Running best"
    )

    # 1.0x is no change, and it is where the eye should return to: everything
    # below it is a patch that made something slower.
    ax.axhline(BASELINE, color="#999999", lw=1, ls="--", zorder=1)
    ax.annotate(
        "baseline 1.00x",
        (0, BASELINE),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9,
        color="#666666",
    )

    if others:
        grey_x, grey_y = _points(others)
        ax.scatter(grey_x, grey_y, s=18, color="#cccccc", zorder=3, label="No improvement")
    # Drawn apart and in red: a patch with a large mean and a slower input is
    # exactly the failure the input set was added to catch, so it should be
    # visible on the plot rather than hidden among the grey.
    if slower:
        red_x, red_y = _points(slower)
        ax.scatter(red_x, red_y, s=30, color=red, marker="x", zorder=4, label="Slower on an input")
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
    # The labels lean up and to the right of their point, so the axes need room
    # on both or the last and most interesting one is clipped. The best attempt
    # is usually the last, which is exactly the corner that runs out of space.
    if plotted:
        last = max(r.number for r in plotted)
        ax.set_xlim(-0.5, last + (0.22 * last + 1 if labels and records else 0.5))
        top = max(r.speedup for r in plotted if r.speedup is not None)
        ax.set_ylim(top=top * (3.0 if labels and records else 1.2))
    ax.set_xlabel("Attempt #")
    ax.set_ylabel("Speedup, geometric mean over inputs (higher is better)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    handles, names = ax.get_legend_handles_labels()
    order = ["No improvement", "Slower on an input", "New best", "Running best"]
    pairs = sorted(zip(handles, names, strict=True), key=lambda h: order.index(h[1]))
    ax.legend([h for h, _ in pairs], [n for _, n in pairs], loc="lower right")
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
    names = input_names(attempts)
    print(render_table(table, names))

    if args.csv:
        write_csv(Path(args.csv), table, names)
        print(f"\ntable: {args.csv}")
    if not args.no_plot:
        out = Path(args.out) if args.out else run / "progress.png"
        records = sum(1 for r in table if r.record)
        title = f"Autoresearch Progress: {len(table)} attempts, {records} improvements ({run.name})"
        plot(table, out, title, log=not args.linear, labels=not args.no_labels)
        print(f"plot:  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
