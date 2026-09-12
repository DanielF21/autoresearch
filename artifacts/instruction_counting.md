# Instruction counting: what it measured, and why it was retired

Retired 2026-09-12, after check 3 (`runs/t1_w4`, 3 rounds at width 4, 12 attempts).

Phase 1 measurement D established that cachegrind counts instructions deterministically
and cheaply, and the referee was built to count every patch. Check 3 measured what that
cost in practice and what the number was worth. It cost 62.8% of the run and nothing
consumed it. This file is what the instrument taught before it was removed; the code is
gone, the finding is not.

---

## 1. What it cost

Where all 22,189 box seconds of check 3 went, from `profile_run.py`:

```
  referee instruction counts, setup, cleanup     13937s   62.8%
                                worker model      3620s   16.3%
                                worker tools      3032s   13.7%
                         referee timing base       711s    3.2%
                          referee full suite       699s    3.2%
                      referee timing patched        66s    0.3%
                          worker box prepare        42s    0.2%
                        referee module tests        29s    0.1%
                              referee verify        25s    0.1%
                              referee canary        17s    0.1%
                           worker box create         7s    0.0%
                              worker harness         3s    0.0%
                           total box seconds     22189s
```

Six of twelve attempts recorded `"ir": null` after exhausting the 1800 s `IR_TIMEOUT`.
Those six account for roughly 11,500 of the 13,937 seconds and produced no number.

## 2. Why it got expensive: the instrument is worst at exactly the patches worth finding

Cachegrind costs about 10.5x on interpreted Python (measurement D). From attempt 0007 on,
every winning patch takes a dense linear algebra path. `runs/t1_w4/attempts/0007/patch.diff`:

```python
Wf = W.astype(_numpy.float32)
C = Wf @ Wf
directed_triangles_arr = _numpy.einsum('ij,ij->i', Wf, C)
```

That is a 1000x1000x1000 float32 GEMM, about 2 GFLOP, which native AVX2 with four threads
finishes in roughly 0.02 s. Valgrind serializes guest threads and emulates every SIMD
instruction in its JIT, so the same call costs on the order of 1,100 s to count. Roughly
50,000x, against 10.5x for the interpreted baseline.

The cost of measuring rises with the quality of the patch.

### The timeout arithmetic

Derived from the two distinct failure signatures in the profile. `IR_TIMEOUT` was 1800 and
the referee counted the base tree first.

| attempts | derived block | decomposition |
|---|---|---|
| 7, 8, 9, 11 | 1866 to 1869 s | base both calls 68 s, then patched `--calls 1` hit 1800 |
| 10, 12 | 2968 to 2972 s | base 68 s, patched `--calls 1` finished in ~1100 s, patched `--calls 2` hit 1800 |

So a full count of a vectorized patch costs at least 3,300 s for the cheaper pair (0010,
0012) and more than 5,400 s for the rest, where `--calls 1` alone exceeded 1800 s. **No
vectorized count was ever allowed to finish, so the upper bound is unmeasured.**

The budget that was in place was the worst available choice: it paid 2,900 s per attempt,
88% of what a successful count costs, and returned nothing.

## 3. The finding worth keeping: instruction count stopped tracking wall clock

Time speedup against instruction reduction, for the six attempts that produced a count:

```
 att   speedup  ir ratio     gap
0001     2.386     2.571    0.93x
0002    19.542     9.503    2.06x
0003    19.509     9.485    2.06x
0004    21.029     9.954    2.11x
0005    30.120    14.212    2.12x
0006    24.085    10.945    2.20x
```

Attempt 0001 tracks. Every bitset patch from 0002 on runs about 2.1x faster than its
instruction reduction predicts, and the gap widens. The base does Python set intersections
that allocate and miss cache; the bitset path is contiguous native integer work on CPython
`long` objects. Similar instruction counts, very different cost per instruction.

This is the substantive result:

> **Instruction count answers "is the patch doing less work", not "is the patch faster".
> The two diverge as soon as a patch changes instruction mix rather than instruction count,
> and every patch worth finding does exactly that.**

Measurement D's determinism claim was not wrong. Ir spread was 0.00002%, against a timing
noise floor of 1.06%, so the count really can resolve a 0.1% change that six timing pairs
cannot. It simply never had to: the wins in this target ran from 2.386x to 78.835x.

## 4. Why it was not moved off the critical path instead

The obvious repair was to fan counting out onto separate boxes after the run, which is what
Phase 1 originally prescribed (`measurements.md`, measurement B and D sections). Instruction
counts are immune to contention, so they can be packed many to a box without corruption,
and at 64 to 128 concurrent counts a full 672 attempt pilot would finish in hours rather
than blocking every round barrier.

That was costed and rejected. The reason is section 3, not the cost: the number was not
informing the agents (zero of twelve rationales mention it, one reasoning trace cites it
descriptively and then reasons entirely from time), and it is a biased proxy for the thing
the experiment measures. A cheaper way to collect a misleading number is not an improvement.

## 5. What was removed

`guest/count_ir.py`, the `--calls` mode of `guest/time_target.py`, `IrCounts` and
`Measurement.ir` in `types.py`, `IR_TIMEOUT` and the counting step in `referee/referee.py`,
the `ir` column in the worker's history index, and `valgrind` from the box image.
`util-linux` stays, because `taskset` pins every timing launch to core 2.

Recorded measurements are never rewritten: every `measurement.json` under `runs/` keeps its
`ir` block. `Measurement.from_dict` reads the key with `.get`, so old records still load
with the field ignored.

`phase1/` is untouched. Measurement D measured the cost of instruction counting and that
measurement is still true; what changed is the judgement about whether the number earns it.
