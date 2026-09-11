"""The timed workloads.

Pure Python integer arithmetic, deliberately. What is being characterized is
CPython bytecode dispatch under scheduling noise, which is what the real
networkx benchmarks are dominated by. A sleep would measure the scheduler and a
numpy call would measure BLAS.
"""

from __future__ import annotations

# Fixed forever. The canary's iteration count must NEVER change or it stops
# being comparable across boxes and across time, which is its only job.
CANARY_N = 2_000_000
CANARY_VERSION = "canary_v1"


def work(n: int) -> int:
    x = 0
    for i in range(n):
        x = (x * 31 + i) & 0xFFFFFFFF
    return x


def canary() -> int:
    return work(CANARY_N)
