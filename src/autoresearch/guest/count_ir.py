"""Instruction count of the target on one tree, via cachegrind. Prints one JSON object.

Runs ``time_target.py --calls N`` under valgrind's cachegrind tool with cache
simulation off, so only the instruction count is collected. ASLR is disabled
with ``setarch -R`` and the hash seed is fixed, which measurement D showed makes
the count repeatable to a few parts in ten million.

With ``--calls 1`` and ``--calls 2`` the difference is the per call count and
the remainder is fixed startup cost: interpreter, import, graph construction.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

IR_RE = re.compile(r"==\d+==\s+I\s+refs:\s+([\d,]+)")


def parse_ir(stderr: str) -> int | None:
    """The total instruction count from cachegrind's summary on stderr."""
    m = IR_RE.search(stderr)
    return int(m.group(1).replace(",", "")) if m else None


def command(
    python: str, script: Path, root: str, graph: str, call: str, calls: int, hot: str
) -> list[str]:
    return [
        "setarch",
        platform.machine(),
        "-R",
        "valgrind",
        "--tool=cachegrind",
        "--cache-sim=no",
        "--branch-sim=no",
        "--cachegrind-out-file=/dev/null",
        python,
        str(script),
        "--root",
        root,
        "--graph",
        graph,
        "--call",
        call,
        "--hot",
        hot,
        "--calls",
        str(calls),
        "--no-counters",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--call", required=True)
    ap.add_argument("--hot", default="networkx/algorithms/cluster.py")
    ap.add_argument("--calls", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    script = Path(__file__).parent / "time_target.py"
    argv = command(sys.executable, script, args.root, args.graph, args.call, args.calls, args.hot)
    env = dict(os.environ, PYTHONHASHSEED="0")
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=args.timeout, env=env, check=False
        )
    except subprocess.TimeoutExpired:
        print(json.dumps({"kind": "ir", "error": "timeout", "calls": args.calls}))
        return 2
    ir = parse_ir(proc.stderr)
    inner = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            inner = json.loads(line)
    out = {
        "kind": "ir",
        "calls": args.calls,
        "root": args.root,
        "rc": proc.returncode,
        "ir": ir,
        "result_fp": inner.get("result_fp"),
        "error": "" if ir is not None and proc.returncode == 0 else proc.stderr[-2000:],
    }
    print(json.dumps(out))
    return 0 if ir is not None and proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
