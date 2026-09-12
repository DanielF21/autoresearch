"""The canary: a fixed pure Python workload timed before every referee measurement.

It is the same loop measurement A and B timed. Its only job is to be comparable
across boxes and across time, so its iteration count must never change. A slow
canary means the box is busy or degraded, and the referee records it next to
the timing pairs so a strange result can be explained later.

Prints one JSON object.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

# Fixed forever. See the module docstring.
CANARY_N = 2_000_000
CANARY_VERSION = "canary_v1"


def work(n: int) -> int:
    x = 0
    for i in range(n):
        x = (x * 31 + i) & 0xFFFFFFFF
    return x


def canary() -> int:
    return work(CANARY_N)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    gc.freeze()
    work(min(CANARY_N, 200_000))
    samples = []
    for _ in range(args.repeats):
        gc.collect()
        t0 = time.perf_counter()
        canary()
        samples.append(time.perf_counter() - t0)
    print(
        json.dumps(
            {
                "kind": "canary",
                "version": CANARY_VERSION,
                "n": CANARY_N,
                "repeats": args.repeats,
                "samples": samples,
                "min_s": min(samples),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
