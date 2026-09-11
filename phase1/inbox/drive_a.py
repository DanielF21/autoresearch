"""Measurement A orchestrator. Runs INSIDE the sailbox.

Launching 180 subprocesses from the laptop would pay a network round trip per
launch, so the orchestration happens in the box and only the results come back.

Writes JSONL incrementally so a dropped connection loses nothing.

Protocol (from the plan):
  - two anchors, ~50ms and ~1s, to fit spread = a + b*T
  - m = 11 in-process repeats, take the min
  - k = 30 process launches, take the median of mins
  - 3 replication sets spaced across the window, to separate drift from noise
  - a fixed-N canary beside every set, as a yardstick comparable across boxes
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

# A fixed set of seeds, used identically across arms so comparisons are paired
# on seed. Never a single seed: that overfits to one dict iteration order, and
# in networkx dict order sets graph traversal order.
HASH_SEEDS = ["0", "1", "2", "3", "4"]


def launch(n: int, repeats: int, label: str, seed: str, pin: int | None) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    argv = [sys.executable, str(HERE / "bench_worker.py"),
            "--n", str(n), "--repeats", str(repeats), "--label", label]
    if pin is not None:
        argv = ["taskset", "-c", str(pin)] + argv
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode != 0:
        return {"label": label, "error": r.stderr[-2000:], "returncode": r.returncode}
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    rec["seed"] = seed
    rec["pin"] = pin
    return rec


def calibrate(pin: int | None) -> tuple[int, int, float]:
    """Measure the canary, then size the anchors from its observed rate.

    The canary keeps a FIXED iteration count so it stays comparable forever.
    The anchors are calibrated per box so they actually land near 50ms and 1s,
    which is what the two-point spread fit needs.
    """
    rec = launch(workload.CANARY_N, 5, "calibration", "0", pin)
    t = rec["min_all"]
    rate = workload.CANARY_N / t          # iterations per second
    n_a1 = max(1000, int(rate * 0.050))
    n_a2 = max(1000, int(rate * 1.000))
    return n_a1, n_a2, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30, help="process launches per anchor per set")
    ap.add_argument("--m", type=int, default=11, help="in-process repeats per launch")
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--gap", type=int, default=45, help="idle seconds between sets")
    ap.add_argument("--pin", type=int, default=2, help="taskset core; avoid 0 (IRQ work)")
    ap.add_argument("--out", default="/workspace/results_a.jsonl")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")

    def emit(obj: dict) -> None:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()

    # Block until the box has settled. A fresh box runs ext4lazyinit, which
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
    emit({"kind": "settle", "waited_s": waited,
          "psi_after": provenance.counters()["psi"].get("some_avg10")})
    print(f"settled after {waited}s", flush=True)

    facts = provenance.static_facts()
    emit({"kind": "static_facts", "facts": facts})
    print(f"cpu={facts['cpu']['model_name']} flag_hash={facts['cpu']['flag_hash']} "
          f"vuln_hash={facts['mitigations']['vuln_hash']}", flush=True)

    n_a1, n_a2, cal_t = calibrate(args.pin)
    emit({"kind": "calibration", "canary_n": workload.CANARY_N, "canary_t": cal_t,
          "n_a1": n_a1, "n_a2": n_a2})
    print(f"canary {workload.CANARY_N} iters = {cal_t:.4f}s -> a1 n={n_a1} a2 n={n_a2}", flush=True)

    anchors = [("a1_50ms", n_a1), ("a2_1s", n_a2)]

    for s in range(args.sets):
        # Canary beside every set. If a later box reads 8% slow on this, every
        # measurement in the set can be normalized or discarded in hindsight,
        # whatever the cause turns out to have been.
        c = launch(workload.CANARY_N, 5, workload.CANARY_VERSION, "0", args.pin)
        emit({"kind": "canary", "set": s, "rec": c})
        print(f"[set {s}] canary min={c['min_all']:.4f}s", flush=True)

        for label, n in anchors:
            t_start = time.time()
            for i in range(args.k):
                seed = HASH_SEEDS[i % len(HASH_SEEDS)]
                rec = launch(n, args.m, label, seed, args.pin)
                rec.update(kind="launch", set=s, i=i)
                emit(rec)
            print(f"[set {s}] {label}: {args.k} launches in {time.time()-t_start:.0f}s", flush=True)

        if s < args.sets - 1:
            time.sleep(args.gap)

    emit({"kind": "done", "finished_at": time.time()})
    fh.close()
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
