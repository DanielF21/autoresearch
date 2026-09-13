#!/usr/bin/env python3
"""Measure a noise floor per benchmark input on one real referee box.

    uv run calibrate.py configs/t1_w4c.toml --rounds 7 --skip er1000_005

Creates one referee box, makes two worktrees of the base commit, and times one
against the other the way the referee times a patch against the base. Nothing is
patched, so every ratio that comes back is what a change of nothing looks like.
The floor for an input is the ratio a median of ``[referee].pairs`` such ratios
exceeds only once in a hundred, bootstrapped from the ratios measured here.

No model is called. The cost is the box and the launches.

Inputs need no floor to be calibrated, which is the point: a new target's
config has none until this has run. ``autoresearch check`` comes first, since
calibrating an input the referee could not time would measure nothing.

Prints a line per input ready to paste into the config, and writes every pair to
a JSONL beside the report so the floor can be recomputed without another box.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

from autoresearch import env
from autoresearch.config import load_config
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
) -> tuple[str, dict[str, object]]:
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="the config whose inputs are calibrated")
    p.add_argument("--rounds", type=int, default=7, help="passes of [referee].pairs per input")
    p.add_argument("--skip", action="append", default=[], help="an input to leave out; repeatable")
    p.add_argument("--out", default="artifacts/calibration", help="where the record goes")
    p.add_argument("--keep", action="store_true", help="leave the box running")
    args = p.parse_args(argv)

    from autoresearch.boxes.sail_box import SailBoxFactory
    from autoresearch.referee.referee import Referee

    env.load_dotenv()
    cfg = load_config(Path(args.config))
    only = tuple(i.name for i in cfg.target.inputs if i.name not in args.skip)
    if not only:
        print("every input was skipped", file=sys.stderr)
        return 2
    per_input = args.rounds * cfg.referee.pairs
    print(
        f"calibrating {len(only)} inputs at {per_input} pairs each "
        f"({args.rounds} rounds of {cfg.referee.pairs}), floor at 1 in "
        f"{int(1 / FALSE_ALARM)}: {', '.join(only)}",
        flush=True,
    )

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes = SailBoxFactory(cfg)
    box = boxes.create(name=f"calibrate-{ts}", role="referee")
    print(f"referee box {box.name} ({box.box_id})", flush=True)

    def progress(name: str, done: int) -> None:
        print(f"  {name}: {done}/{per_input} pairs", flush=True)

    try:
        ref = Referee(box, cfg)
        ref.setup()
        measured = ref.null_pairs(rounds=args.rounds, only=only, progress=progress)
        if ref.broken:
            print(f"referee marked broken: {ref.broken}", file=sys.stderr)
    finally:
        if not args.keep:
            box.terminate()
            print("box terminated", flush=True)

    lines: list[str] = []
    records: list[dict[str, object]] = []
    for name in only:
        line, rec = report(name, measured[name], cfg.referee.pairs, FALSE_ALARM)
        lines.append(line)
        records.append(rec)
    print("\n" + "\n".join(lines))

    jsonl = out_dir / f"{ts}.jsonl"
    with jsonl.open("w") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "run",
                    "config": args.config,
                    "sha": cfg.target.sha,
                    "box_id": box.box_id,
                    "rounds": args.rounds,
                    "pairs": cfg.referee.pairs,
                    "repeats_per_launch": cfg.referee.repeats_per_launch,
                    "false_alarm": FALSE_ALARM,
                    "at": ts,
                }
            )
            + "\n"
        )
        for rec in records:
            fh.write(json.dumps({"kind": "floor", **rec}) + "\n")
        for name, pairs in measured.items():
            for pair in pairs:
                fh.write(json.dumps({"kind": "pair", "input": name, **pair.to_dict()}) + "\n")
    print(f"\nrecord: {jsonl}")
    print("\npaste into the config:")
    for rec in records:
        if rec["floor"] is not None:
            print(
                f"noise_floor = {rec['floor']:.4f}   # calibrated {ts}, {rec['clean']} null pairs"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
