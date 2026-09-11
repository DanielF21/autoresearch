"""Measurement D, scoped to cost. Runs INSIDE the box.

Question: how much time does instruction counting add per candidate?

Every candidate will get both an instruction count and a wall clock timing, so
the only open question is price. Four configurations, each run with the
benchmark body executed once and twice. The difference between the two runs
isolates the per-body cost from the fixed cost of starting Python, importing
networkx and building the graph under instrumentation.

  plain      no instrumentation, the reference
  nulgrind   valgrind with no tool: the irreducible cost of valgrind itself
  cg_nosim   cachegrind counting instructions only
  cg_sim     cachegrind also simulating the cache. Slower, but its miss counts
             are the deterministic signal for memory layout changes, which an
             instruction count cannot see.

ASLR is disabled and the hash seed fixed for these runs, because determinism is
the goal here, the opposite of the wall clock runs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import provenance  # noqa: E402

SHA = "c94928ed9489"
URL = "https://github.com/networkx/networkx"
BENCHES = {"long": (1000, 0.05), "short": (1000, 0.01)}
ROOT = "/workspace/nx_a"
RX = re.compile(r"==\d+==\s+(I\s+refs|I1\s+misses|LLi\s+misses|D\s+refs|D1\s+misses|"
                r"LLd\s+misses|LL\s+refs|LL\s+misses):\s+([\d,]+)")
CONFIGS = {
    "plain": [],
    "nulgrind": ["valgrind", "--tool=none"],
    "cg_nosim": ["valgrind", "--tool=cachegrind", "--cache-sim=no", "--branch-sim=no",
                 "--cachegrind-out-file=/tmp/cg.%p"],
    "cg_sim": ["valgrind", "--tool=cachegrind", "--cache-sim=yes", "--branch-sim=no",
               "--cachegrind-out-file=/tmp/cg.%p"],
}


def sh(cmd: str, timeout: int = 900) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def run(cfg: str, bench: str, reps: int, pin) -> dict:
    n, p = BENCHES[bench]
    env = dict(os.environ, PYTHONHASHSEED="0")
    inner = [sys.executable, str(HERE / "null_worker.py"), "--root", ROOT, "--n", str(n),
             "--p", str(p), "--repeats", str(reps), "--label", f"d:{cfg}", "--no-counters"]
    argv = ["setarch", platform.machine(), "-R"] + CONFIGS[cfg] + inner
    if pin is not None:
        argv = ["taskset", "-c", str(pin)] + argv
    t0 = time.monotonic()
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=3600)
    wall = time.monotonic() - t0
    counts = {re.sub(r"\s+", " ", k): int(v.replace(",", "")) for k, v in RX.findall(r.stderr)}
    try:
        inner_rec = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        inner_rec = {}
    return {"cfg": cfg, "bench": bench, "reps": reps, "wall_s": wall, "rc": r.returncode,
            "counts": counts, "inner_min_s": inner_rec.get("min_all"),
            "result_fp": inner_rec.get("result_fp"), "edges": inner_rec.get("edges"),
            "err": (r.stderr[-800:] if r.returncode else "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/workspace/out.jsonl")
    args = ap.parse_args()

    fh = open(args.out, "w")

    def emit(o: dict) -> None:
        fh.write(json.dumps(o) + "\n")
        fh.flush()

    waited = 0
    while waited < 180:
        psi = provenance.counters()["psi"].get("some_avg10", 0.0)
        busy = subprocess.run("ps aux | grep -c '[e]xt4lazyinit'", shell=True,
                              capture_output=True, text=True).stdout.strip()
        if psi < 0.5 and busy == "0":
            break
        time.sleep(5)
        waited += 5
    emit({"kind": "static_facts", "facts": provenance.static_facts()})

    rc, out = sh(f"git clone -q --depth 50 {URL} {ROOT} && cd {ROOT} && git checkout -q {SHA}")
    if rc:
        emit({"kind": "fatal", "stage": "clone", "out": out})
        return 2
    sh("pip uninstall -y networkx")
    _, vg = sh("valgrind --version")
    emit({"kind": "setup", "valgrind": vg.strip()})

    benches = ["short"] if args.smoke else ["short", "long"]
    cfgs = ["plain", "cg_nosim"] if args.smoke else list(CONFIGS)
    # Warm the .pyc cache so no configuration pays compilation.
    run("plain", "short", 1, args.pin)

    for bench in benches:
        for cfg in cfgs:
            for reps in (1, 2):
                rec = run(cfg, bench, reps, args.pin)
                rec["kind"] = "d_run"
                emit(rec)
                print(f"[D] {bench} {cfg} x{reps}: {rec['wall_s']:.1f}s "
                      f"Ir={rec['counts'].get('I refs')}", flush=True)

    if not args.smoke:
        # Determinism check: the whole argument for instruction counts.
        for i in range(2):
            rec = run("cg_nosim", "short", 1, args.pin)
            rec.update(kind="d_det", i=i)
            emit(rec)

    emit({"kind": "done", "finished_at": time.time()})
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
