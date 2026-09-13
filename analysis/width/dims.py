"""The seven dimensions of the width analysis: one table and one figure each.

Every function takes the loaded runs and an output directory, writes
``dN_<name>.csv`` and ``dN_<name>.png`` there, and returns the table it wrote so a
test can read it. Speedups are on a log axis, as in ``analyze.py``. Dollar figures
use the list price table in ``orchestrator/status.py`` and are labelled as that
table's bracket, never as a measured bill.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.width.load import Row, Run
from autoresearch.orchestrator.status import PRICES_PER_M

Table = list[dict[str, Any]]
COLORS = {1: "#1f77b4", 4: "#ff7f0e", 16: "#2ca02c", 64: "#d62728"}
OUTCOME_ORDER = (
    "record",
    "real speedup",
    "below the noise floor",
    "slower on an input",
    "timing failed",
    "wrong result",
    "tests failed",
    "out of scope",
    "did not apply",
    "not measured yet",
    "no patch",
)


def _color(width: int) -> str:
    return COLORS.get(width, "#7f7f7f")


def _write(path: Path, table: Table) -> None:
    if not table:
        path.write_text("")
        return
    keys: list[str] = []
    for row in table:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)


def _fmt(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _best_after(run: Run, upto: int, strict: bool = True) -> float | None:
    """Best cleared geomean through a round.

    Strict counts only fully timed attempts under the current rule; not strict is
    what the run itself believed, from the flag the referee wrote at the time.
    """
    best: float | None = None
    for r in run.rows:
        cleared = (r.clears_noise and r.complete) if strict else r.clears_seen
        if r.round <= upto and cleared and r.speedup is not None:
            best = r.speedup if best is None else max(best, r.speedup)
    return best


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----- D1 headline ---------------------------------------------------------------------


def d1_headline(runs: list[Run], out: Path) -> Table:
    """Best real speedup after each round, against rounds and against attempts."""
    table: Table = []
    for run in runs:
        cum = 0
        for n in range(1, run.rounds_done + 1):
            cum += len(run.in_round(n))
            table.append(
                {
                    "run_id": run.run_id,
                    "width": run.width,
                    "round": n,
                    "attempts_cum": cum,
                    "best": _fmt(_best_after(run, n)),
                    "best_as_run_saw_it": _fmt(_best_after(run, n, strict=False)),
                }
            )
    _write(out / "d1_headline.csv", table)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        rows = [t for t in table if t["run_id"] == run.run_id and t["best"] is not None]
        label = f"width {run.width}"
        for ax, key in ((axes[0], "round"), (axes[1], "attempts_cum")):
            ax.step(
                [t[key] for t in rows],
                [t["best"] for t in rows],
                where="post",
                color=_color(run.width),
                label=label,
                marker="o",
                ms=3,
            )
            seen = [t for t in rows if t["best_as_run_saw_it"] != t["best"]]
            if seen:
                ax.step(
                    [t[key] for t in rows],
                    [t["best_as_run_saw_it"] for t in rows],
                    where="post",
                    color=_color(run.width),
                    ls=":",
                    lw=1,
                    label=f"{label}, as the run saw it (an input untimed)",
                )
    axes[0].set_xlabel("round")
    axes[1].set_xlabel("cumulative attempts")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("best real speedup, geomean over 5 inputs, all 5 timed")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title("D1: best by round")
    axes[1].set_title("D1: best by attempts spent")
    _save(fig, out / "d1_headline.png")
    return table


# ----- D2 stall ------------------------------------------------------------------------


def d2_stall(runs: list[Run], out: Path) -> Table:
    """When each width last set a record, and how big each step was."""
    steps: Table = []
    summary: Table = []
    for run in runs:
        prev: float | None = None
        for r in run.records():
            assert r.speedup is not None
            steps.append(
                {
                    "run_id": run.run_id,
                    "width": run.width,
                    "round": r.round,
                    "number": r.number,
                    "speedup": _fmt(r.speedup),
                    "step": _fmt(r.speedup / prev) if prev else None,
                    "functions": " ".join(r.functions),
                    "strategies": " ".join(r.strategies),
                }
            )
            prev = r.speedup
        recs = run.records()
        by_round = Counter(r.round for r in recs)
        last = recs[-1] if recs else None
        summary.append(
            {
                "run_id": run.run_id,
                "width": run.width,
                "rounds_done": run.rounds_done,
                "records": len(recs),
                "records_as_run_saw_it": sum(1 for r in run.rows if r.record_seen),
                "incomplete_cleared": sum(1 for r in run.rows if r.clears_seen and not r.complete),
                "last_record_round": last.round if last else None,
                "rounds_since_record": run.rounds_done - last.round if last else None,
                "last_record_attempt": last.number if last else None,
                "final_best": _fmt(last.speedup) if last else None,
                "records_by_round": " ".join(
                    str(by_round.get(n, 0)) for n in range(1, run.rounds_done + 1)
                ),
            }
        )
    _write(out / "d2_stall.csv", summary)
    _write(out / "d2_stall_records.csv", steps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    nruns = max(1, len(runs))
    for i, run in enumerate(runs):
        by_round = Counter(r.round for r in run.records())
        xs = [n + (i - nruns / 2) * 0.8 / nruns for n in range(1, run.rounds_done + 1)]
        axes[0].bar(
            xs,
            [by_round.get(n, 0) for n in range(1, run.rounds_done + 1)],
            width=0.8 / nruns,
            color=_color(run.width),
            label=f"width {run.width}",
        )
        mine = [s for s in steps if s["run_id"] == run.run_id and s["step"] is not None]
        axes[1].scatter(
            [s["round"] for s in mine],
            [s["step"] for s in mine],
            color=_color(run.width),
            label=f"width {run.width}",
        )
    axes[0].set_title("D2: records set per round")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("new records")
    axes[1].set_title("D2: size of each record step")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("new best / previous best")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    _save(fig, out / "d2_stall.png")
    return summary


# ----- D3 diversity --------------------------------------------------------------------


def _round_diversity(run: Run, n: int) -> dict[str, Any]:
    batch = run.in_round(n)
    patched = [r for r in batch if r.has_patch]
    funcs = Counter(f for r in patched for f in r.functions)
    strats = Counter(s for r in patched for s in r.strategies)
    pairs = list(combinations(patched, 2))
    top_f = funcs.most_common(1)[0] if funcs else ("", 0)
    top_s = strats.most_common(1)[0] if strats else ("", 0)
    hashes = {r.norm_hash for r in patched}
    return {
        "run_id": run.run_id,
        "width": run.width,
        "round": n,
        "attempts": len(batch),
        "with_patch": len(patched),
        "unique_diffs": len(hashes),
        "duplicate_share": _fmt(1 - len(hashes) / len(patched)) if patched else None,
        "distinct_functions": len(funcs),
        "top_function": top_f[0],
        "top_function_share": _fmt(top_f[1] / len(patched)) if patched else None,
        "distinct_strategies": len(strats),
        "top_strategy": top_s[0],
        "top_strategy_share": _fmt(top_s[1] / len(patched)) if patched else None,
        "mean_jaccard_functions": _fmt(
            _mean([_jaccard(a.functions, b.functions) for a, b in pairs])
        ),
        "mean_jaccard_strategies": _fmt(
            _mean([_jaccard(a.strategies, b.strategies) for a, b in pairs])
        ),
    }


def d3_diversity(runs: list[Run], out: Path) -> Table:
    """How alike the attempts of one round are, and how that moves with history."""
    table: Table = [_round_diversity(run, n) for run in runs for n in range(1, run.rounds_done + 1)]
    _write(out / "d3_diversity.csv", table)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for run in runs:
        mine = [t for t in table if t["run_id"] == run.run_id]
        label = f"width {run.width}"
        kw = {"color": _color(run.width), "label": label, "marker": "o", "ms": 3}
        xs = [t["round"] for t in mine]
        js = [t["mean_jaccard_functions"] for t in mine]
        if any(j is not None for j in js):
            axes[0].plot(xs, [j if j is not None else float("nan") for j in js], **kw)
        axes[1].plot(xs, [t["top_function_share"] for t in mine], **kw)
        axes[2].plot(xs, [t["distinct_strategies"] for t in mine], **kw)
    axes[0].set_title("D3: mean pairwise Jaccard of functions touched")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("D3: share of attempts on the most touched function")
    axes[1].set_ylim(0, 1.05)
    axes[2].set_title("D3: distinct strategy classes (heuristic)")
    for ax in axes:
        ax.set_xlabel("round")
        ax.grid(True, alpha=0.3)
        ax.legend()
    _save(fig, out / "d3_diversity.png")
    return table


# ----- D4 exploitation -----------------------------------------------------------------


def _split(rows: list[Row]) -> dict[str, Any]:
    building = [r for r in rows if r.builds_on_record]
    fresh = [r for r in rows if r.builds_on_record is False]

    def rate(group: list[Row], key: str) -> float | None:
        return _fmt(sum(1 for r in group if getattr(r, key)) / len(group)) if group else None

    def med(group: list[Row]) -> float | None:
        vals = [r.speedup for r in group if r.speedup is not None]
        return _fmt(statistics.median(vals)) if vals else None

    return {
        "with_record_available": len(rows),
        "building": len(building),
        "share_building": _fmt(len(building) / len(rows)) if rows else None,
        "mean_overlap": _fmt(_mean([r.overlap for r in rows if r.overlap is not None])),
        "cleared_rate_building": rate(building, "clears_noise"),
        "cleared_rate_fresh": rate(fresh, "clears_noise"),
        "record_rate_building": rate(building, "record"),
        "record_rate_fresh": rate(fresh, "record"),
        "median_geomean_building": med(building),
        "median_geomean_fresh": med(fresh),
    }


def d4_exploitation(runs: list[Run], out: Path) -> Table:
    """Whether attempts build on the record they saw, and whether that pays."""
    per_round: Table = []
    summary: Table = []
    for run in runs:
        for n in range(1, run.rounds_done + 1):
            rows = [r for r in run.in_round(n) if r.overlap is not None]
            per_round.append({"run_id": run.run_id, "width": run.width, "round": n, **_split(rows)})
        rows = [r for r in run.rows if r.overlap is not None]
        summary.append({"run_id": run.run_id, "width": run.width, **_split(rows)})
    _write(out / "d4_exploitation.csv", summary)
    _write(out / "d4_exploitation_rounds.csv", per_round)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        mine = [t for t in per_round if t["run_id"] == run.run_id]
        axes[0].plot(
            [t["round"] for t in mine],
            [
                t["share_building"] if t["share_building"] is not None else float("nan")
                for t in mine
            ],
            color=_color(run.width),
            label=f"width {run.width}",
            marker="o",
            ms=3,
        )
    axes[0].set_title("D4: share of attempts building on the record (overlap >= 0.5)")
    axes[0].set_xlabel("round")
    axes[0].set_ylim(0, 1.05)
    xs = list(range(len(summary)))
    axes[1].bar(
        [x - 0.2 for x in xs],
        [s["record_rate_building"] or 0 for s in summary],
        width=0.4,
        label="building on the record",
        color="#555555",
    )
    axes[1].bar(
        [x + 0.2 for x in xs],
        [s["record_rate_fresh"] or 0 for s in summary],
        width=0.4,
        label="fresh",
        color="#bbbbbb",
    )
    axes[1].set_xticks(xs, [f"width {s['width']}" for s in summary])
    axes[1].set_title("D4: share of attempts that set a record")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    _save(fig, out / "d4_exploitation.png")
    return summary


# ----- D5 outcomes ---------------------------------------------------------------------


def _outcome_class(r: Row) -> str:
    return "record" if r.record else r.outcome_class


def d5_outcomes(runs: list[Run], out: Path) -> Table:
    """Where the attempts went: outcome classes and stop reasons, per width and round."""
    table: Table = []
    stops: Table = []
    for run in runs:
        for n in range(1, run.rounds_done + 1):
            batch = run.in_round(n)
            counts = Counter(_outcome_class(r) for r in batch)
            table.append(
                {
                    "run_id": run.run_id,
                    "width": run.width,
                    "round": n,
                    "attempts": len(batch),
                    **{k: counts.get(k, 0) for k in OUTCOME_ORDER},
                }
            )
        by_stop = Counter(r.stop_reason for r in run.rows)
        empty = sum(1 for r in run.rows if r.has_patch and r.rationale_chars == 0)
        stops.append(
            {
                "run_id": run.run_id,
                "width": run.width,
                "attempts": len(run.rows),
                **dict(sorted(by_stop.items())),
                "patch_without_rationale": empty,
                "mean_turns": _fmt(_mean([float(r.turns) for r in run.rows])),
            }
        )
    _write(out / "d5_outcomes.csv", table)
    _write(out / "d5_stop_reasons.csv", stops)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    xs = list(range(len(runs)))
    bottom = [0.0] * len(runs)
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(OUTCOME_ORDER):
        shares = []
        for run in runs:
            n = len(run.rows)
            shares.append(sum(1 for r in run.rows if _outcome_class(r) == cls) / n if n else 0)
        if any(shares):
            axes[0].bar(xs, shares, bottom=bottom, label=cls, color=cmap(i % 10))
            bottom = [b + s for b, s in zip(bottom, shares, strict=True)]
    axes[0].set_xticks(xs, [f"width {run.width}" for run in runs])
    axes[0].set_title("D5: outcome of every attempt")
    axes[0].legend(fontsize=7)
    bottom = [0.0] * len(runs)
    reasons = sorted({r.stop_reason for run in runs for r in run.rows})
    for i, reason in enumerate(reasons):
        shares = [
            sum(1 for r in run.rows if r.stop_reason == reason) / max(1, len(run.rows))
            for run in runs
        ]
        axes[1].bar(xs, shares, bottom=bottom, label=reason, color=cmap(i % 10))
        bottom = [b + s for b, s in zip(bottom, shares, strict=True)]
    axes[1].set_xticks(xs, [f"width {run.width}" for run in runs])
    axes[1].set_title("D5: why the worker stopped")
    axes[1].legend(fontsize=7)
    for run in runs:
        mine = [t for t in table if t["run_id"] == run.run_id]
        axes[2].plot(
            [t["round"] for t in mine],
            [(t["real speedup"] + t["record"]) / t["attempts"] for t in mine],
            color=_color(run.width),
            label=f"width {run.width}",
            marker="o",
            ms=3,
        )
    axes[2].set_title("D5: share of attempts that were real speedups")
    axes[2].set_xlabel("round")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    _save(fig, out / "d5_outcomes.png")
    return table


# ----- D6 cost -------------------------------------------------------------------------


def _dollars(run: Run, prompt: int, cached: int, completion: int) -> tuple[float, float]:
    prices = PRICES_PER_M.get(run.model)
    if prices is None:
        return 0.0, 0.0
    comp = completion * prices[1]
    fresh = max(0, prompt - cached)
    return (fresh * prices[0] + comp) / 1e6, (prompt * prices[0] + comp) / 1e6


def d6_cost(runs: list[Run], out: Path) -> Table:
    """Tokens, wall time and list price dollars per attempt and per record."""
    per_round: Table = []
    summary: Table = []
    for run in runs:
        cum_low = cum_high = 0.0
        cum_wall = 0.0
        for rec in run.rounds:
            n = len(rec.attempt_numbers)
            u = rec.usage
            low, high = _dollars(run, u.prompt_tokens, u.cached_tokens, u.completion_tokens)
            cum_low += low
            cum_high += high
            cum_wall += rec.worker_wall_s + rec.referee_wall_s
            per_round.append(
                {
                    "run_id": run.run_id,
                    "width": run.width,
                    "round": rec.round,
                    "attempts": n,
                    "prompt_per_attempt": round(u.prompt_tokens / n) if n else None,
                    "cached_share": _fmt(u.cached_tokens / u.prompt_tokens)
                    if u.prompt_tokens
                    else None,
                    "completion_per_attempt": round(u.completion_tokens / n) if n else None,
                    "worker_wall_s": round(rec.worker_wall_s),
                    "referee_wall_s": round(rec.referee_wall_s),
                    "round_wall_s": round(rec.worker_wall_s + rec.referee_wall_s),
                    "dollars_low": _fmt(low),
                    "dollars_high": _fmt(high),
                    "cum_dollars_low": _fmt(cum_low),
                    "cum_dollars_high": _fmt(cum_high),
                    "cum_wall_h": _fmt(cum_wall / 3600),
                    "best": _fmt(_best_after(run, rec.round)),
                }
            )
        records = len(run.records())
        n = len(run.rows)
        summary.append(
            {
                "run_id": run.run_id,
                "width": run.width,
                "attempts": n,
                "records": records,
                "dollars_low": _fmt(cum_low),
                "dollars_high": _fmt(cum_high),
                "dollars_per_attempt_low": _fmt(cum_low / n) if n else None,
                "dollars_per_record_low": _fmt(cum_low / records) if records else None,
                "dollars_per_record_high": _fmt(cum_high / records) if records else None,
                "wall_h": _fmt(cum_wall / 3600),
                "wall_h_per_record": _fmt(cum_wall / 3600 / records) if records else None,
                "final_best": _fmt(_best_after(run, run.rounds_done)),
                "price_table": "orchestrator/status.py PRICES_PER_M list price; low reads "
                "cached tokens as free, high at full input price",
            }
        )
    _write(out / "d6_cost.csv", summary)
    _write(out / "d6_cost_rounds.csv", per_round)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for run in runs:
        mine = [t for t in per_round if t["run_id"] == run.run_id]
        kw = {"color": _color(run.width), "label": f"width {run.width}", "marker": "o", "ms": 3}
        axes[0].plot([t["round"] for t in mine], [t["prompt_per_attempt"] for t in mine], **kw)
        pts = [t for t in mine if t["best"] is not None]
        axes[2].step(
            [t["cum_dollars_low"] for t in pts], [t["best"] for t in pts], where="post", **kw
        )
    axes[0].set_title("D6: prompt tokens per attempt")
    axes[0].set_xlabel("round")
    xs = list(range(len(runs)))
    worker = [
        statistics.mean(r.worker_wall_s for r in run.rounds) if run.rounds else 0 for run in runs
    ]
    referee = [
        statistics.mean(r.referee_wall_s for r in run.rounds) if run.rounds else 0 for run in runs
    ]
    axes[1].bar(xs, worker, label="worker phase", color="#555555")
    axes[1].bar(xs, referee, bottom=worker, label="referee phase", color="#bbbbbb")
    axes[1].set_xticks(xs, [f"width {run.width}" for run in runs])
    axes[1].set_title("D6: mean wall seconds per round")
    axes[2].set_title("D6: best against cumulative dollars (low bracket, list price)")
    axes[2].set_xlabel("cumulative dollars, cached tokens read as free")
    axes[2].set_yscale("log")
    if any(t["cum_dollars_low"] for t in per_round):
        axes[2].set_xscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    _save(fig, out / "d6_cost.png")
    return summary


# ----- D7 records ----------------------------------------------------------------------


def d7_records(runs: list[Run], out: Path) -> Table:
    """What the records look like: spread across inputs, margin from the gate, size."""
    table: Table = []
    for run in runs:
        names = run.input_names
        for r in run.records():
            ratios = [v for v in r.per_input.values() if v is not None and v > 0]
            logs = [math.log(v) for v in ratios]
            margins = [
                v * r.floors[name] - 1
                for name, v in r.per_input.items()
                if v is not None and name in r.floors
            ]
            table.append(
                {
                    "run_id": run.run_id,
                    "width": run.width,
                    "round": r.round,
                    "number": r.number,
                    "speedup": _fmt(r.speedup),
                    "worst": _fmt(r.worst),
                    "log_var": _fmt(statistics.pvariance(logs)) if len(logs) > 1 else None,
                    "gate_margin": _fmt(min(margins)) if margins else None,
                    "changed_lines": r.changed_lines,
                    "files": len(r.files),
                    "functions": " ".join(r.functions),
                    "strategies": " ".join(r.strategies),
                    **{name: _fmt(r.per_input.get(name)) for name in names},
                }
            )
    _write(out / "d7_records.csv", table)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    names = runs[0].input_names if runs else []
    nruns = max(1, len(runs))
    for i, run in enumerate(runs):
        recs = run.records()
        if not recs:
            continue
        final = recs[-1]
        xs = [j + (i - nruns / 2) * 0.8 / nruns for j in range(len(names))]
        axes[0].bar(
            xs,
            [final.per_input.get(n) or float("nan") for n in names],
            width=0.8 / nruns,
            color=_color(run.width),
            label=f"width {run.width}, attempt {final.number}",
        )
        kw = {"color": _color(run.width), "label": f"width {run.width}", "marker": "o", "ms": 4}
        axes[1].plot([r.round for r in recs], [r.changed_lines for r in recs], **kw)
        mine = [t for t in table if t["run_id"] == run.run_id and t["gate_margin"] is not None]
        axes[2].plot([t["round"] for t in mine], [t["gate_margin"] for t in mine], **kw)
    axes[0].set_xticks(list(range(len(names))), names, rotation=20)
    axes[0].set_yscale("log")
    axes[0].set_title("D7: the final record's speedup on each input")
    axes[1].set_title("D7: changed lines of each record")
    axes[1].set_xlabel("round")
    axes[2].set_title("D7: worst input's margin above the regression gate")
    axes[2].set_xlabel("round")
    axes[2].axhline(0, color="black", lw=0.8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, out / "d7_records.png")
    return table


def run_all(runs: list[Run], out: Path) -> dict[str, Table]:
    out.mkdir(parents=True, exist_ok=True)
    return {
        "d1": d1_headline(runs, out),
        "d2": d2_stall(runs, out),
        "d3": d3_diversity(runs, out),
        "d4": d4_exploitation(runs, out),
        "d5": d5_outcomes(runs, out),
        "d6": d6_cost(runs, out),
        "d7": d7_records(runs, out),
    }
