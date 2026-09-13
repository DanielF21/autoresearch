"""Load the width runs, write every dimension's table and figure, print a short summary.

    uv run python -m analysis.width runs/t1_wx1 runs/t1_wx4 runs/t1_wx16 runs/t1_wx64 \
        --out report_materials/width --source runs/intake/networkx/repo
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from pathlib import Path

from analysis.width import dims, load


def _sha(run_dir: Path) -> str:
    data = tomllib.loads((run_dir / "config.toml").read_text())
    return str(data.get("target", {}).get("sha", ""))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run directories, one per width")
    p.add_argument("--out", default="report_materials/width")
    p.add_argument(
        "--source",
        default="runs/intake/networkx/repo",
        help="a clone holding the target's base commit, for function attribution",
    )
    p.add_argument("--attempts-csv", default="", help="also write every row here")
    args = p.parse_args(argv)

    runs: list[load.Run] = []
    for d in args.runs:
        run_dir = Path(d)
        source = load.git_source(Path(args.source), _sha(run_dir))
        run = load.load_run(run_dir, source)
        runs.append(run)
        print(
            f"{run.run_id}: width {run.width}, {run.rounds_done} of {run.rounds_total} rounds, "
            f"{len(run.rows)} attempts, {len(run.records())} records, "
            f"best {(run.records()[-1].speedup or 0):.3f}x"
            if run.records()
            else f"{run.run_id}: width {run.width}, {run.rounds_done} rounds, no record"
        )
    runs.sort(key=lambda r: r.width)
    out = Path(args.out)
    tables = dims.run_all(runs, out)
    if args.attempts_csv:
        rows = [
            {k: (" ".join(v) if isinstance(v, tuple) else v) for k, v in vars(r).items()}
            for run in runs
            for r in run.rows
        ]
        with Path(args.attempts_csv).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
