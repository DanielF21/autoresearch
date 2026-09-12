#!/usr/bin/env python3
"""Where the time went, for every part of a run.

    uv run profile_run.py runs/t1_w4
    uv run profile_run.py runs/t1_w4 --out artifacts/t1_w4_time.png --csv time.csv

Three levels, printed as tables and drawn as one figure:

  orchestrator  round wall clock, split into the worker phase and the referee
                phase, plus what the round barrier wasted waiting for the
                slowest of the four.
  worker        measured exactly from transcript.jsonl: box create, box
                prepare, every model call, every tool call by name, harness.
  referee       part measured, part modelled, part derived. See below.

The referee records only its total ``wall_s`` plus test durations and the
timings themselves, so its breakdown cannot all be read off. What each number
is:

  measured   module tests, full suite, and the per launch benchmark times.
  modelled   a timing launch costs one fixed startup plus ``repeats_per_launch``
             bodies. Fixed is 0.3 s, measurement D, long bench, plain config.
             The same shape covers the two verify launches and the canary.
  derived    worktree setup, cleanup and guest overhead is whatever is left.

The derived block is named "setup and cleanup" and is a residual, not a claim
about any one step. Everything labelled measured is read straight from the run
directory.

A run recorded before instruction counting was retired carries that cost inside
the referee wall clock, so its residual is large and its error list says why.
See artifacts/instruction_counting.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch import history
from autoresearch.config import RunConfig, load_config
from autoresearch.types import Attempt, Measurement

# One process launch pays this before the first call of the target: interpreter
# start, importing networkx from the tree, building the graph. Measurement D,
# long bench, plain config: per candidate 1.7 s of which fixed is 0.3 s.
LAUNCH_FIXED_S = 0.3

WORKER_ORDER = ["box create", "box prepare", "model", "tools", "harness"]
REFEREE_ORDER = [
    "module tests",
    "full suite",
    "verify",
    "canary",
    "timing base",
    "timing patched",
    "setup and cleanup",
]
COLOURS = {
    "box create": "#8fb8de",
    "box prepare": "#5b8db8",
    "model": "#2f6f9f",
    "tools": "#f0a35e",
    "harness": "#cfcfcf",
    "module tests": "#a8d5a2",
    "full suite": "#4bbf73",
    "verify": "#2f7d4f",
    "canary": "#d9d36a",
    "timing base": "#e2725b",
    "timing patched": "#f2a58d",
    "setup and cleanup": "#7b5ea7",
}


@dataclass
class WorkerProfile:
    """Every second of one attempt's worker phase, measured from the transcript."""

    number: int
    wall_s: float
    parts: dict[str, float] = field(default_factory=dict)
    tools: dict[str, float] = field(default_factory=dict)
    tool_calls: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    model_calls: int = 0
    slowest_model_s: float = 0.0
    slowest_tool: tuple[str, float] = ("", 0.0)


@dataclass
class RefereeProfile:
    number: int
    wall_s: float
    parts: dict[str, float] = field(default_factory=dict)
    launches: int = 0
    errors: int = 0


def worker_profile(attempt: Attempt, transcript: str) -> WorkerProfile:
    """Split the worker's wall clock using the timestamps already on every record."""
    recs = [json.loads(line) for line in transcript.splitlines() if line.strip()]
    p = WorkerProfile(number=attempt.ref.number, wall_s=attempt.wall_s)
    if not recs:
        p.parts["harness"] = attempt.wall_s
        return p

    # ``Attempt`` does not carry the turn count; the transcript's end record does.
    end = next((r for r in recs if r["kind"] == "end"), recs[-1])
    p.turns = int(end.get("turns", 0))
    start = float(end["t"]) - attempt.wall_s
    box = next((r for r in recs if r["kind"] == "box"), None)

    model_s = 0.0
    tools: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    first_model_start: float | None = None
    prev_t = float(box["t"]) if box else start

    for r in recs:
        if r["kind"] == "model":
            latency = float(r.get("latency_s", 0.0))
            model_s += latency
            p.model_calls += 1
            p.slowest_model_s = max(p.slowest_model_s, latency)
            if first_model_start is None:
                first_model_start = float(r["t"]) - latency
            prev_t = float(r["t"])
        elif r["kind"] == "tool":
            # A tool runs between the record before it and its own record.
            d = max(0.0, float(r["t"]) - prev_t)
            name = str(r.get("name", "?"))
            tools[name] += d
            calls[name] += 1
            if d > p.slowest_tool[1]:
                p.slowest_tool = (name, d)
            prev_t = float(r["t"])

    # Every part is clamped at zero. These are bar heights, and a disagreement
    # between the recorded wall clock and the timestamps must not draw a
    # negative bar; it lands in harness as an unexplained remainder instead.
    p.parts["box create"] = max(0.0, (float(box["t"]) - start) if box else 0.0)
    p.parts["box prepare"] = max(
        0.0, (first_model_start - float(box["t"])) if box and first_model_start else 0.0
    )
    p.parts["model"] = model_s
    p.parts["tools"] = sum(tools.values())
    p.parts["harness"] = max(0.0, attempt.wall_s - sum(p.parts.values()))
    p.tools = dict(tools)
    p.tool_calls = dict(calls)
    return p


def referee_profile(number: int, m: Measurement, config: RunConfig) -> RefereeProfile:
    """Measured where the record has it, modelled for launches, derived for the rest."""
    repeats = config.referee.repeats_per_launch
    p = RefereeProfile(number=number, wall_s=m.wall_s)
    tests = {t.scope: t.duration_s for t in m.tests}

    base_timing = sum(LAUNCH_FIXED_S + repeats * x.base_s for x in m.pairs)
    patched_timing = sum(LAUNCH_FIXED_S + repeats * x.patched_s for x in m.pairs)
    # Two verify launches, one call each under cProfile, one per tree.
    one = m.pairs[0] if m.pairs else None
    verify = 2 * LAUNCH_FIXED_S + ((one.base_s + one.patched_s) if one else 0.0)
    canary = LAUNCH_FIXED_S + 5 * (m.canary_s or 0.0)

    p.parts["module tests"] = tests.get("module", 0.0)
    p.parts["full suite"] = tests.get("full", 0.0)
    p.parts["verify"] = verify
    p.parts["canary"] = canary
    p.parts["timing base"] = base_timing
    p.parts["timing patched"] = patched_timing
    p.parts["setup and cleanup"] = max(0.0, m.wall_s - sum(p.parts.values()))
    p.launches = 2 * len(m.pairs) + 2 + 1  # timing, verify, canary
    p.errors = len(m.errors)
    return p


@dataclass
class RoundProfile:
    round: int
    worker_wall_s: float
    referee_wall_s: float
    worker_each: list[float]
    referee_each: list[float]

    @property
    def wall_s(self) -> float:
        return self.worker_wall_s + self.referee_wall_s

    @property
    def worker_idle_s(self) -> float:
        """Box seconds spent waiting at the barrier for the slowest worker."""
        return sum(self.worker_wall_s - w for w in self.worker_each)

    @property
    def referee_idle_s(self) -> float:
        return sum(self.referee_wall_s - r for r in self.referee_each)


def build(
    run: Path,
) -> tuple[RunConfig, list[WorkerProfile], list[RefereeProfile], list[RoundProfile]]:
    paths = history.RunPaths(run)
    config = load_config(paths.config)
    attempts = history.load_history(paths)
    rounds = history.read_rounds(paths)

    workers: list[WorkerProfile] = []
    referees: list[RefereeProfile] = []
    for a in attempts:
        t = paths.attempts / a.ref.dirname / "transcript.jsonl"
        workers.append(worker_profile(a, t.read_text() if t.exists() else ""))
        if a.measurement is not None:
            referees.append(referee_profile(a.ref.number, a.measurement, config))

    by_number = {w.number: w for w in workers}
    ref_by_number = {r.number: r for r in referees}
    rps = [
        RoundProfile(
            round=r.round,
            worker_wall_s=r.worker_wall_s,
            referee_wall_s=r.referee_wall_s,
            worker_each=[by_number[n].wall_s for n in r.attempt_numbers if n in by_number],
            referee_each=[
                ref_by_number[n].wall_s for n in r.measured_numbers if n in ref_by_number
            ],
        )
        for r in rounds
    ]
    return config, workers, referees, rps


# ----- reporting -------------------------------------------------------------------------


def _bar(value: float, total: float, width: int = 24) -> str:
    n = 0 if total <= 0 else round(width * value / total)
    return "#" * n + "." * (width - n)


def render(
    config: RunConfig,
    workers: list[WorkerProfile],
    referees: list[RefereeProfile],
    rounds: list[RoundProfile],
) -> str:
    out: list[str] = []
    run_wall = sum(r.wall_s for r in rounds)

    out.append(
        f"= orchestrator: {len(rounds)} rounds, width {config.width}, {run_wall:,.0f}s total"
    )
    out.append("")
    out.append(
        f"{'round':>5} {'wall':>7} {'worker':>8} {'referee':>8} "
        f"{'w idle':>8} {'r idle':>8}  share of run"
    )
    for r in rounds:
        out.append(
            f"{r.round:>5} {r.wall_s:>7.0f} {r.worker_wall_s:>8.0f} {r.referee_wall_s:>8.0f} "
            f"{r.worker_idle_s:>8.0f} {r.referee_idle_s:>8.0f}  {_bar(r.wall_s, run_wall)}"
        )
    tot_w = sum(r.worker_wall_s for r in rounds)
    tot_r = sum(r.referee_wall_s for r in rounds)
    out.append(
        f"{'all':>5} {run_wall:>7.0f} {tot_w:>8.0f} {tot_r:>8.0f} "
        f"{sum(r.worker_idle_s for r in rounds):>8.0f} {sum(r.referee_idle_s for r in rounds):>8.0f}"
        f"   worker {100 * tot_w / run_wall:.0f}%, referee {100 * tot_r / run_wall:.0f}%"
    )

    out += ["", "= worker, measured from transcript.jsonl", ""]
    head = f"{'att':>4} {'wall':>7} {'turns':>6} " + " ".join(f"{k:>12}" for k in WORKER_ORDER)
    out.append(head)
    for w in workers:
        cells = " ".join(f"{w.parts.get(k, 0.0):>12.1f}" for k in WORKER_ORDER)
        out.append(f"{w.number:>4} {w.wall_s:>7.0f} {w.turns:>6} {cells}")
    totals = {k: sum(w.parts.get(k, 0.0) for w in workers) for k in WORKER_ORDER}
    out.append(
        f"{'sum':>4} {sum(w.wall_s for w in workers):>7.0f} {'':>6} "
        + " ".join(f"{totals[k]:>12.0f}" for k in WORKER_ORDER)
    )

    out += ["", "  worker time inside the tools, by tool", ""]
    names = sorted({n for w in workers for n in w.tools})
    out.append(f"{'tool':>16} {'seconds':>10} {'calls':>8} {'s per call':>11}  share of tool time")
    tool_total = sum(sum(w.tools.values()) for w in workers)
    for n in names:
        s = sum(w.tools.get(n, 0.0) for w in workers)
        c = sum(w.tool_calls.get(n, 0) for w in workers)
        out.append(f"{n:>16} {s:>10.0f} {c:>8} {s / max(1, c):>11.2f}  {_bar(s, tool_total)}")

    out += ["", "= referee, measured + modelled + derived", ""]
    out.append(f"{'att':>4} {'wall':>7} " + " ".join(f"{k[:12]:>13}" for k in REFEREE_ORDER))
    for ref in referees:
        cells = " ".join(f"{ref.parts.get(k, 0.0):>13.1f}" for k in REFEREE_ORDER)
        flag = f"  {ref.errors} error(s)" if ref.errors else ""
        out.append(f"{ref.number:>4} {ref.wall_s:>7.0f} {cells}{flag}")
    rtot = {k: sum(ref.parts.get(k, 0.0) for ref in referees) for k in REFEREE_ORDER}
    out.append(
        f"{'sum':>4} {sum(ref.wall_s for ref in referees):>7.0f} "
        + " ".join(f"{rtot[k]:>13.0f}" for k in REFEREE_ORDER)
    )

    out += ["", "= box seconds for the whole run, every category", ""]
    every = {f"worker {k}": v for k, v in totals.items()}
    every.update({f"referee {k}": v for k, v in rtot.items()})
    grand = sum(every.values())
    for k, v in sorted(every.items(), key=lambda kv: -kv[1]):
        out.append(f"{k:>44} {v:>9.0f}s {100 * v / grand:>6.1f}%  {_bar(v, grand)}")
    out.append(f"{'total box seconds':>44} {grand:>9.0f}s")
    return "\n".join(out)


def write_csv(path: Path, workers: list[WorkerProfile], referees: list[RefereeProfile]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["side", "attempt", "category", "seconds"])
        for p in workers:
            for k, v in p.parts.items():
                w.writerow(["worker", p.number, k, round(v, 3)])
            for k, v in p.tools.items():
                w.writerow(["worker_tool", p.number, k, round(v, 3)])
        for r in referees:
            for k, v in r.parts.items():
                w.writerow(["referee", r.number, k, round(v, 3)])


def plot(
    config: RunConfig,
    workers: list[WorkerProfile],
    referees: list[RefereeProfile],
    rounds: list[RoundProfile],
    out: Path,
    run_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(19, 15))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.5, 1.5, 1.1], hspace=0.42, wspace=0.16)
    run_wall = sum(r.wall_s for r in rounds)
    fig.suptitle(
        f"Where the time went: {run_name}, {len(rounds)} rounds at width {config.width}, "
        f"{run_wall / 60:.0f} min wall clock",
        fontsize=15,
    )

    # A. round timeline, worker phase then referee phase, laid out as it happened
    ax = fig.add_subplot(gs[0, :])
    t = 0.0
    for r in rounds:
        ax.barh(0, r.worker_wall_s, left=t, color="#2f6f9f", edgecolor="white")
        ax.text(
            t + r.worker_wall_s / 2,
            0,
            f"w{r.round}\n{r.worker_wall_s:.0f}s",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
        )
        t += r.worker_wall_s
        ax.barh(0, r.referee_wall_s, left=t, color="#7b5ea7", edgecolor="white")
        ax.text(
            t + r.referee_wall_s / 2,
            0,
            f"referee {r.round}\n{r.referee_wall_s:.0f}s",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
        )
        t += r.referee_wall_s
    ax.set_yticks([])
    ax.set_xlabel("seconds from the start of round 1")
    ax.set_title(
        "A. The run as it happened: worker phase then referee phase, per round", loc="left"
    )
    ax.set_xlim(0, t)

    # B. referee per attempt, full scale, so a runaway block is not hidden
    ax = fig.add_subplot(gs[1, :])
    xs = [ref.number for ref in referees]
    bottom = [0.0] * len(referees)
    for k in REFEREE_ORDER:
        vals = [ref.parts.get(k, 0.0) for ref in referees]
        ax.bar(xs, vals, bottom=bottom, color=COLOURS[k], label=k, width=0.72, edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    for ref in referees:
        if ref.errors:
            ax.text(
                ref.number,
                ref.wall_s + 40,
                f"{ref.errors}\nerror" + ("s" if ref.errors > 1 else ""),
                ha="center",
                fontsize=7,
                color="#7b5ea7",
            )
    ax.set_xticks(xs)
    ax.set_xlabel("attempt")
    ax.set_ylabel("seconds")
    ax.set_title("B. Referee, every attempt, true scale", loc="left")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # C. the same bars with the residual cut away, so the measured steps are legible
    ax = fig.add_subplot(gs[2, 0])
    bottom = [0.0] * len(referees)
    for k in REFEREE_ORDER[:-1]:
        vals = [ref.parts.get(k, 0.0) for ref in referees]
        ax.bar(xs, vals, bottom=bottom, color=COLOURS[k], label=k, width=0.72, edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    ax.set_xticks(xs)
    ax.set_xlabel("attempt")
    ax.set_ylabel("seconds")
    ax.set_title("C. Referee without the residual: what the measured steps cost", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # D. worker per attempt
    ax = fig.add_subplot(gs[2, 1])
    wxs = [w.number for w in workers]
    bottom = [0.0] * len(workers)
    for k in WORKER_ORDER:
        vals = [w.parts.get(k, 0.0) for w in workers]
        ax.bar(wxs, vals, bottom=bottom, color=COLOURS[k], label=k, width=0.72, edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
    ax.set_xticks(wxs)
    ax.set_xlabel("attempt")
    ax.set_ylabel("seconds")
    ax.set_title("D. Worker, every attempt, measured from the transcript", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    # E. every category over the whole run, box seconds
    ax = fig.add_subplot(gs[3, :])
    every: dict[str, float] = {}
    for k in WORKER_ORDER:
        every[f"worker: {k}"] = sum(w.parts.get(k, 0.0) for w in workers)
    for k in REFEREE_ORDER:
        every[f"referee: {k}"] = sum(ref.parts.get(k, 0.0) for ref in referees)
    items = sorted(every.items(), key=lambda kv: kv[1])
    grand = sum(every.values())
    colours = [COLOURS[k.split(": ", 1)[1]] for k, _ in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=colours, edgecolor="white")
    for i, (_, v) in enumerate(items):
        ax.text(v + grand * 0.004, i, f"{v:,.0f}s  {100 * v / grand:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("box seconds across the whole run")
    ax.set_title(
        f"E. Every category, whole run. {grand:,.0f} box seconds of work "
        f"inside {run_wall:,.0f}s of wall clock",
        loc="left",
    )
    ax.set_xlim(0, max(every.values()) * 1.18)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", help="a run directory, for example runs/t1_w4")
    p.add_argument("--out", default="", help="figure path; default <run>/time.png")
    p.add_argument("--csv", default="", help="also write every category as csv")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    run = Path(args.run)
    if not (run / "attempts").is_dir():
        print(f"{run} does not look like a run directory", file=sys.stderr)
        return 2
    config, workers, referees, rounds = build(run)
    if not rounds:
        print(f"no completed rounds in {run}", file=sys.stderr)
        return 1
    print(render(config, workers, referees, rounds))
    if args.csv:
        write_csv(Path(args.csv), workers, referees)
        print(f"\ncsv:    {args.csv}")
    if not args.no_plot:
        out = Path(args.out) if args.out else run / "time.png"
        plot(config, workers, referees, rounds, out, run.name)
        print(f"figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
