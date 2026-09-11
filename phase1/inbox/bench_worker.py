"""One process launch: run a workload m times, report the min plus provenance.

Min over in-process repeats is correct here. Within a single process the code,
heap layout, dict ordering and hash seed are all fixed, so variation really is
one-sided additive interference. The median across launches (done by the
orchestrator) handles the level where variation is real rather than noise.

Prints one JSON object to stdout.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance  # noqa: E402
import workload  # noqa: E402


def time_once(n: int) -> tuple[float, dict]:
    # Start every repeat from an identical GC phase. Do NOT disable GC: that
    # would hide the benefit of a patch that reduces allocations, which is
    # exactly a class of patch the harness exists to find.
    gc.collect()
    before = provenance.counters()
    t0 = time.perf_counter()
    workload.work(n)
    t1 = time.perf_counter()
    after = provenance.counters()
    return t1 - t0, provenance.delta(before, after)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="loop iterations")
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--label", default="anchor")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    gc.freeze()  # module-level heap out of gen2 scans

    for _ in range(args.warmup):
        workload.work(min(args.n, 200_000))

    samples = []
    for _ in range(args.repeats):
        t, d = time_once(args.n)
        contaminated, reasons = provenance.is_contaminated(d)
        samples.append({"t": t, "delta": d, "contaminated": contaminated, "reasons": reasons})

    clean = [s["t"] for s in samples if not s["contaminated"]]
    out = {
        "label": args.label,
        "n": args.n,
        "repeats": args.repeats,
        "pid": os.getpid(),
        "hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "cpu": provenance.sched_getcpu(),
        # raw, every sample, never pre-aggregated
        "samples": samples,
        "min_clean": min(clean) if clean else None,
        "min_all": min(s["t"] for s in samples),
        "n_clean": len(clean),
        "n_contaminated": len(samples) - len(clean),
    }
    json.dump(out, sys.stdout)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
