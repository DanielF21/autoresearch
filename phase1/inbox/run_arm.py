"""Run ONE arm of measurement B inside the referee box. Prints JSONL.

Single-arm by design. The laptop orchestrates arm ordering and cross-box load,
so no flag files and no polling the referee's filesystem mid-measurement.

In-box load is NOT a spin loop. Memory bandwidth, last-level-cache pressure and
page-cache churn matter far more than ALU contention for this workload, and a
spin loop produces none of them. 3 workers, not 6: this box has 4 vCPU (Phase 0),
so 6 would measure saturation rather than realistic worker load.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import provenance  # noqa: E402
import workload  # noqa: E402

HASH_SEEDS = ["0", "1", "2", "3", "4"]

LOAD_SRC = """
import sys, random
random.seed(int(sys.argv[1]))
buf = [bytearray(1 << 20) for _ in range(96)]   # ~96 MiB; LLC is 33 MiB
i = 0
while True:
    d = {}
    for i in range(40000):
        d[random.randrange(1 << 20)] = i
    for b in buf:
        b[random.randrange(len(b) - 8)] = i & 0xFF
    del d
"""


def launch(n, repeats, label, seed, pin):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    argv = [sys.executable, str(HERE / "bench_worker.py"),
            "--n", str(n), "--repeats", str(repeats), "--label", label]
    if pin is not None:
        argv = ["taskset", "-c", str(pin)] + argv
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode != 0:
        return {"label": label, "error": r.stderr[-1500:], "returncode": r.returncode}
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    rec.update(seed=seed, pin=pin)
    return rec


def start_load(n, avoid_cpu, nproc):
    cpus = [c for c in range(nproc) if c != avoid_cpu] or [0]
    return [subprocess.Popen(
        ["taskset", "-c", str(cpus[i % len(cpus)]), sys.executable, "-c", LOAD_SRC, str(i)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["facts", "calibrate", "arm"], default="arm")
    ap.add_argument("--arm", default="B0")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--n", type=int, default=0, help="anchor iterations (from calibrate)")
    ap.add_argument("--pin", type=int, default=2)
    ap.add_argument("--inbox-load", type=int, default=3)
    ap.add_argument("--settle", type=int, default=15)
    ap.add_argument("--cycle", type=int, default=0)
    args = ap.parse_args()

    def emit(o):
        print(json.dumps(o), flush=True)

    if args.mode == "facts":
        # Block until the box has settled: a fresh box runs ext4lazyinit, which
        # churns the disk and pushes PSI above the contamination threshold.
        waited = 0
        while waited < 180:
            psi = provenance.counters()["psi"].get("some_avg10", 0.0)
            busy = subprocess.run("ps aux | grep -c '[e]xt4lazyinit'", shell=True,
                                  capture_output=True, text=True).stdout.strip()
            if psi < 0.5 and busy == "0":
                break
            time.sleep(5)
            waited += 5
        emit({"kind": "settle", "waited_s": waited})
        emit({"kind": "static_facts", "facts": provenance.static_facts()})
        return 0

    if args.mode == "calibrate":
        rec = launch(workload.CANARY_N, 5, "calibration", "0", args.pin)
        rate = workload.CANARY_N / rec["min_all"]
        emit({"kind": "calibration", "canary_t": rec["min_all"],
              "n_anchor": max(1000, int(rate * 1.000))})
        return 0

    # arm mode
    nproc = os.cpu_count() or 4
    procs = []
    if args.arm == "B2c":
        procs = start_load(args.inbox_load, args.pin, nproc)
        time.sleep(args.settle)   # cgroup accounting and page cache need to settle

    c = launch(workload.CANARY_N, 5, workload.CANARY_VERSION, "0", args.pin)
    emit({"kind": "canary", "arm": args.arm, "cycle": args.cycle, "rec": c})

    c0 = provenance.counters()
    t0 = time.time()
    for i in range(args.k):
        rec = launch(args.n, args.m, args.arm, HASH_SEEDS[i % len(HASH_SEEDS)], args.pin)
        rec.update(kind="launch", arm=args.arm, cycle=args.cycle, i=i)
        emit(rec)
    c1 = provenance.counters()
    emit({"kind": "arm_summary", "arm": args.arm, "cycle": args.cycle,
          "wall_s": time.time() - t0, "counters": provenance.delta(c0, c1),
          "steal_delta": c1["steal"] - c0["steal"],
          "psi_after": c1["psi"].get("some_avg10")})

    for p in procs:
        p.kill()
    for p in procs:
        p.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
