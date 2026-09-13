# Measurements

Raw output in `runs/`. This file records what each number means and what it decided.

---

## Phase 0: sailbox characterization

Box `phase0-20260911-100411`, size `m`, taken 2026-09-11. Raw: `runs/phase0/20260911-100411.md`.

### The box

| Property | Value |
|---|---|
| vCPU | **4** (not 8) |
| Memory | 7954 MiB usable, though `memory_limit_gib=16` was accepted and reported back as 16384 |
| Arch | amd64, Intel Xeon @ 2.50GHz, microcode masked to `0x1` |
| Kernel | 6.1.155 |
| Python | 3.12.13, same build as the local uv interpreter |
| valgrind | 3.19.0 |
| cgroup | **v1** (tmpfs at `/sys/fs/cgroup`, controllers `blkio cpu cpuacct cpuset devices freezer memory pids`) |

`size="m"` yields 4 vCPU, and `memory_limit_gib` behaves as a cap rather than a request. Anything sizing the worker pool off the documented "8 to 128 GiB" range is wrong.

### Decision-relevant readings

| Question | Reading | Consequence |
|---|---|---|
| **Is the PMU exposed?** | `perf stat -e instructions,cycles` returns `<not supported>` for both. `perf_event_paranoid = 2` | **No.** Hardware instruction counts are unavailable, so the near free alternative to cachegrind is off the table. Measurement D proceeds as designed and the cheap gate has no cheaper substitute. |
| **Dedicated cores or throttled quota?** | `cpu.cfs_quota_us = -1`, `cfs_period_us = 100000`, `nr_throttled = 0`, `throttled_time = 0` | **Unlimited, no quota.** The throttling confound is absent. The per run `nr_throttled` guard stays in, but it is expected to read zero and its job is to catch a change in platform policy. |
| **SMT?** | `smt/active = 0`. All four `thread_siblings_list` entries are singletons (`0`, `1`, `2`, `3`) | **No hyperthreading.** Arm B2a (contend the referee core's SMT sibling) is not runnable and is deleted. The referee does not need to reserve a sibling, which frees a core. |
| **Oversubscribed host?** | Steal (field 9 of `/proc/stat`) = 0 at rest | No steal observed at idle. Not yet tested under load; that is experiment B. |
| **Is the clock trustworthy?** | `constant_tsc`, `nonstop_tsc` present; clocksource = `tsc` | **Yes.** `time.perf_counter` is reliable. No escalation needed. |
| **Can turbo or governor be controlled?** | `/sys/devices/system/cpu/cpu0/cpufreq/` does not exist | **No**, as expected in a guest. Do not spend time trying. Frequency drift is mitigated by ABBA interleaving, not by control. |
| **Is the box quiet?** | PSI `some avg10` was **1.03 at boot**, settling to **0.00** once `ext4lazyinit` exited. Load average 0.24 → 0.06 | Quiet once settled. **A freshly created box is not measurable for roughly the first minute.** |

### What this changes in the protocol

1. **`ext4lazyinit` must exit before any timing begins.** A fresh box runs lazy ext4 inode table init, which churns the disk and pushes PSI `some avg10` above the 1.0 contamination threshold. Measuring immediately after create would have contaminated every early run and looked like platform noise. Now a blocking settle loop in `scripts/phase0_characterize.py`.

2. **All cgroup reads use v1 paths**, with v2 tried first. The plan's `/sys/fs/cgroup/cpu.max` does not exist here. Throttle counters live at `/sys/fs/cgroup/cpu/cpu.stat`.

3. **Arm B2a is deleted** (no SMT to contend).

4. **Arm B2c drops from 6 concurrent worker processes to 3.** Six was written against an assumed 8 vCPU; on 4 vCPU it would leave the referee nothing and measure saturation rather than realistic worker load.

5. **Measurement D is now load bearing.** With no PMU, cachegrind is the only cheap gate available, so if D fails there is no fallback except dropping the gate entirely and buying referee boxes.

### Script bugs this run exposed

- `perf stat ... | tail -6` truncated away the counter lines, which sit above the timing block. The first run looked like perf had succeeded. Fixed to capture full output.
- The cgroup check assumed v2 and silently fell through to a bare two number dump that was easy to misread.

---

## CPU homogeneity across boxes

Run 2026-09-11, `scripts/check_cpu_homogeneity.py`. Raw: `runs/cpu-homogeneity/`.

**Result: 7 boxes across two batches, all identical.**

| Field | Value | Uniform? |
|---|---|---|
| Model | Intel Xeon @ 2.50GHz | yes |
| family/model/stepping | 6/63/2 (**Haswell-EP**, Xeon E5 v3) | yes |
| CPU flag set (md5 of sorted flags) | `f67a2ab0f059` | yes |
| SIMD | avx, avx2, bmi2, fma, sse4_2. **No AVX-512** | yes |
| numpy baseline / dispatch | X86_V2 / X86_V3 | yes |
| vCPU, MHz | 4, 2499.998 | yes |

**Why this matters.** Instruction counts are comparable across boxes only if the same code path executes. CPU *model* is not the determinant; runtime CPUID dispatch is, because numpy selects SIMD kernels that way at import. Identical flag hash and identical numpy dispatch means `Ir` counts from different boxes are directly comparable, so the cheap gate can fan out freely.

**What this does not prove.** Seven boxes at one moment is a snapshot of current placement, not a policy guarantee. Haswell-EP is 2014 silicon; if Sail adds newer hardware, AVX-512 capable boxes would dispatch numpy differently and counts would silently stop being comparable.

**Standing defence, since placement is not controllable.** Record `FLAGHASH` alongside every `Ir` count. It is one line and free. If heterogeneity ever appears, either pin `NPY_DISABLE_CPU_FEATURES` to a common baseline or bucket counts by flag hash and compare only within a bucket. Do not build anything that assumes a uniform fleet.

### Script bug this run exposed

Two, both the same shape: a suppressed error reading as a clean result.

1. The numpy probe imported `numpy.core._multiarray_umath`, which numpy 2.x renamed to `numpy._core`. Wrapped in `2>/dev/null || echo NA`, so the first run reported `NA` and **still printed PASS** on the strength of `/proc/cpuinfo` flags alone. Fixed to try both paths and surface stderr.
2. Cleanup wrapped `terminate()` in a bare `except: pass`. It did not actually fire here, but it would have hidden a real billing leak.

**Platform gotcha worth remembering:** `Sailbox.list()` returns terminated boxes as well as live ones. A name in a listing is not evidence the box is alive. Check `.status` (`running` / `sleeping` / `terminated`) and `.cpu_used_vcpu`.

---

## A. Uncontended timing noise

Box `measure-a-20260911-104158`, 2026-09-11. k=30 launches, m=11 repeats, 3 sets, pinned to core 2, 45s idle gaps. 180 launches, ~19 min. Raw: `runs/measure-a/20260911-104158.jsonl`.

### Verdict: MARGINAL, and the platform drifts 8.7%

**The headline is not the spread. It is the drift.** The canary, whose iteration count is fixed and never changes, fell monotonically across the three sets:

| Set | Canary | vs set 0 |
|---|---|---|
| 0 | 0.2391s | baseline |
| 1 | 0.2247s | **-6.06%** |
| 2 | 0.2183s | **-8.72%** |

Monotone, and the same trend appears inside each set (first third slower than last third) and in the per set wall times (329s, 317s, 295s).

**What is established and what is not.** Established: the box's speed is not stable over a twenty minute window and can move by roughly 9%. That is one box, one session, three checkpoints, and three points falling in order occurs by chance with probability 1/6, so the *direction* is not established. It could be a warmup effect or a random walk that happened to walk one way. Thermal throttling is ruled out only if the direction is real. Cause is not observable from the guest, since there is no `cpufreq` access.

The magnitude and timescale drive every downstream decision; the direction changes only whether a cheap burn-in mitigation exists. Measurement B collects a second drift trace on a different box for free.

**What that would have cost a block design.** Timing a "patch" in set 2 against a baseline from set 0 makes it look **6.08% faster while changing absolutely nothing.** That is larger than most patches the harness will find. Block designs are not merely suboptimal here, they are actively fraudulent.

This settles the ABBA decision with evidence rather than argument, and it settles the "never compare against a cached baseline" rule.

### The three spreads, which are different numbers for different purposes

| Statistic | a1 (46ms) | a2 (898ms) | What it is for |
|---|---|---|---|
| **Within set** `s_rel` | 1.96% | **1.56%** | The instrument's actual noise. **Judge the platform on this.** |
| Pooled across sets | 4.15% | 3.57% | Conflates drift with noise. Meaningless as a quality measure |
| Single adjacent pair, central 95% | +/-5.62% | **+/-4.05%** | What one ABBA pair can resolve |

Contamination exclusion rate was **0.71%** on the long anchor and 0% on the short one, so almost nothing was thrown away and the counters were not doing heavy lifting.

### The operative number for the referee

**A single adjacent pair cannot resolve better than about 4%.** That is one sample against one sample with no averaging, so it is worse than the within set spread of 1.56%. The referee's minimum detectable effect therefore falls roughly as `4% / sqrt(n_pairs)`:

| Pairs per candidate | Approx. resolvable effect | Referee cost per candidate |
|---|---|---|
| 1 | ~4.0% | 1x |
| 5 | ~1.8% | 5x |
| 20 | ~0.9% | 20x |

This is the real trade the project faces, and it is now quantified rather than guessed.

### What this changes

1. **Adjacent paired baselines are mandatory, not preferred.** 8.7% drift over 20 minutes with no cached baseline allowed.
2. **The pairs-per-candidate count is now a tunable with a known curve.** It directly sets both the minimum detectable effect and the referee's throughput.
3. **The canary earned its place on its first run.** It is what made the drift legible; the anchors alone would have looked like noise.
4. **Measurement C is the number that actually governs**, since it measures the adjacent paired ratio through the full referee path including install. A's job was to characterize the instrument, and it did.

### Analysis bug this run exposed

The first pass pooled all 90 launches per anchor and reported `s_rel = 3.57%`, which tripped the `> 3%` rule and printed **FAIL**. That was wrong: it was measuring drift and noise together and blaming the platform. Within set spread is 1.56%, which is MARGINAL. The plan called for 3 spaced sets precisely so drift could be separated from noise, and then the analysis pooled them anyway. Fixed: `analyse()` now reports within set, pooled, drift, and adjacent pair as four separate numbers, and the verdict judges on within set.

## B. Contended timing noise

Box `measure-b-ref-20260911-112247` plus 3 load boxes, 2026-09-11. k=6 per arm instance, 2 instances per cycle (ABBA, order flipped on odd cycles), m=11, 3 cycles. 216 launches, ~25 min. Raw: `runs/measure-b/20260911-112247.jsonl`.

### Results, paired against the B0 measured in the same cycle

| Arm | Anchor inflation | Canary inflation | Per cycle | Spread ratio vs B0 |
|---|---|---|---|---|
| B0 idle | baseline | baseline | | 1.00x |
| **B2c** in box load (3 workers) | **+5.77%** | **+8.55%** | +5.63, +6.62, +5.77 | 0.71x |
| **B1** cross box load (3 boxes) | **+1.65%** | **+2.51%** | +1.25, +2.19, +1.65 | 1.26x |

Both effects are same signed in all 3 cycles. The anchor and the independent canary agree on direction and rough magnitude for both arms.

### 1. In box load corrupts the referee. Settled.

+5.77% on the anchor, +8.55% on the canary, consistent across cycles. **The referee gets a dedicated box**, and the cachegrind fan out gets its own. At roughly $2 per box this was always the likely answer, and it is now evidenced.

> Update 2026-09-12: the dedicated referee box shipped. The cachegrind fan out never did, and instruction counting was removed entirely rather than moved. See `artifacts/instruction_counting.md`.

### 2. Cross box interference is real but small, and probably cancels

+1.65% anchor, +2.51% canary, same sign in 3 of 3 cycles. Boxes are not fully isolated.

**But a uniform inflation cancels in a paired ratio.** With ABBA pairs timed adjacently, both arms sit under the same conditions, so a flat couple of percent affects both equally. What would hurt is *spread* inflation, and that was modest at 1.26x.

**So N referee boxes should still give roughly N times throughput**, which is the finding that decides whether the project has a 10 hour bottleneck or a 1 hour one. Supported, not proven.

### 3. Steal is NOT a usable contention detector. This was the plan's alarm.

| Arm | Steal deltas | Total | As fraction of wall |
|---|---|---|---|
| B0 idle | [0,0,1,1,0,1] | 3 jiffies | 0.0071% |
| B2c | [1,1,1,1,1,1] | 6 jiffies | 0.0143% |
| B1 | [0,1,0,0,0,1] | 2 jiffies | 0.0048% |

**Idle B0 recorded more total steal than B1.** Steal at these magnitudes is background, not signal, and it is blind to an effect the canary picked up without difficulty.

The design's production guard was "record steal on every timing run and auto invalidate anything it fires on." That guard does not work on this platform. **Replace it with the canary**, which detected every effect in this experiment. The steal counter stays recorded, since it is free and would catch a gross change in platform policy, but it is no longer the primary detector.

The automated verdict initially printed CROSS BOX EFFECT DETECTED purely because B1 steal was greater than zero. That rule was wrong. Fixed to compare against the idle arm rather than against zero.

### 4. Drift is not reproducible between boxes

| Box | Canary movement | Monotone? |
|---|---|---|
| Measurement A referee | **-8.72%** over 20 min | yes |
| Measurement B referee | **-2.16%**, settled after the first arm | no |

Both moved in the same direction, which mildly supports a warmup reading, but **the magnitude varies by 4x between boxes.** Drift magnitude is not predictable, so it cannot be calibrated out per box. Adjacent pairing remains the only defence, now for a second independent reason.

### Harness bug found after A and B ran

`provenance.is_contaminated()` checked the key `psi_some_avg10`, but `provenance.delta()` writes it as `psi_some_avg10_after`. The PSI guard therefore never fired in measurement A or B. The other four guards (steal, throttle, major page fault, CPU migration) were live. Per-arm PSI readings in B were 0.00 to 0.33, below the 1.0 threshold, so the dead guard most likely excluded nothing it should have. Fixed before C and D.

## Block C: mergeability calibration

**Dropped for now** (user decision, 2026-09-11). The finding from PR #7971 stands on record: **Done 2026-09-12: the referee now times five shapes and refuses a patch slower on any of them. See `artifacts/generality.md`.** networkx maintainers require performance reported across multiple graph shapes including regressions, while the referee as designed accepts on one shape. Revisit before opening any pull request.

## C. Real noise floor via null patch

Boxes `measure-c0-20260911-124254` and `measure-c1-20260911-124254`, run in parallel, 2026-09-11. networkx `c94928ed9489`. Target `nx.clustering` on `erdos_renyi_graph(1000, p, seed=42, directed=True)`: long p=0.05 (1.39s body), short p=0.01 (80ms body). 12 quartets, ABBA then BAAB, m=7, hash seed shared within a quartet. 24 pairs per bench per box. Raw: `runs/measure-cd/20260911-124254-c0.jsonl`, `-c1.jsonl`.

Verification on every tree: networkx imported from the requested tree, `cluster.py` executed with 0.92 to 0.93 tottime share, results identical across trees (`e46b57123b` long, `b4c0d8b651` short), `.pyc` fresh before import on every measured launch.

| | Median ratio | Per-pair s_rel | Worst false speedup | 90% CI of median | NF (plan formula) |
|---|---|---|---|---|---|
| C0 long | 1.0012 | 0.93% | +1.65% | [0.9950, 1.0035] | 1.0278 |
| C0 short | 1.0033 | 1.35% | +4.49% | **[1.0009, 1.0124]** | 1.0405 |
| C1 long | 1.0009 | 0.88% | +1.77% | [0.9978, 1.0045] | 1.0265 |
| C1 short | 0.9991 | 1.25% | +4.71% | [0.9944, 1.0055] | 1.0374 |

No bias halt fired (all medians within 1% of 1). Exclusions: 7 long samples (steal 6, PSI 1); the PSI guard is live after the fix.

### Resolution curve, pooled C0 and C1 nulls (48 pairs per bench)

95th percentile of the median ratio under the null, by pairs per candidate. A candidate must exceed this to hold its false positive rate under 5%.

| Pairs | Long | Short | Long cost | Short cost |
|---|---|---|---|---|
| 1 | 1.65% | 3.77% | 20.6s | 1.6s |
| 3 | 1.12% | 1.67% | 61.8s | 4.8s |
| 5 | 0.72% | 1.53% | 103s | 8.0s |
| 10 | 0.56% | 1.18% | 206s | 16s |
| 24 | 0.41% | 0.79% | 494s | 38s |

**Resolution per second favours the short bench by about 4x** (s_rel squared times launch cost: 0.89² × 10.3 = 8.2 for long, 1.55² × 0.8 = 1.9 for short). Caveat: the two densities weight the triangle counting inner loop differently, so a patch's speedup need not be equal across them.

### Findings

1. **ABBA pairing cancelled drift.** The canary ranged 8.69% (C0) and 6.61% (C1) during the runs; paired medians stayed within 0.33% of 1.
2. **The whitespace patch is indistinguishable from no patch.** The `.pyc` trap did not materialise: recompilation lands in import, outside the timed region, and the verify step pre-compiled both trees.
3. **1 of 4 null 90% CIs excluded 1.0** (C0 short). Expected: P(at least one of four) ≈ 0.34. Acceptance must use the null threshold table, never "CI excludes 1".
4. **Real code was quieter per pair than measurement A's synthetic loop** (0.89% vs 2.19%). Different boxes and times; cause not established.
5. **Drift direction:** 4 of 4 boxes ended faster (A -8.72%, B -2.16%, C0 -4.08%, C1 -3.05%), p ≈ 1/16 under a symmetric null. C traces not monotone (6 of 11 steps up). Suggestive of warmup plus wander; not established.
6. **The plan's 1.02 epistemic floor is conservative for this target.** At 3 or more pairs the long bench resolves below 2%. Lowering it is a user decision.

Per-candidate wall cost, long bench: 5 pairs = 103s plus tests 61 to 101s, about 3 to 3.5 min. 290 candidates, one referee box: 14 to 17 hours. Parallel referee boxes (measurement B) divide this.

## D. Cost of instruction counting

Scope changed by user decision: every candidate gets both an instruction count and a wall clock timing, so D measures cost only, not gate validity. Box `measure-d-20260911-124254`, 2026-09-11, valgrind 3.19.0. Same target as C. ASLR off (`setarch -R`), `PYTHONHASHSEED=0`, pinned to core 2. Each configuration run with the body executed once and twice; body cost = difference, fixed = 1x minus body. Raw: `runs/measure-cd/20260911-124254-d.jsonl`.

| Bench | Config | Per candidate (1x) | Fixed | Body | Body slowdown |
|---|---|---|---|---|---|
| long | plain | 1.7s | 0.3s | 1.41s | 1.0x |
| long | nulgrind | 11.0s | 4.7s | 6.34s | 4.5x |
| long | **cg_nosim** | **23.8s** | 9.0s | 14.83s | 10.5x |
| long | **cg_sim** | **60.3s** | 20.3s | 39.99s | 28.3x |
| short | plain | 0.3s | 0.3s | 0.07s | 1.0x |
| short | nulgrind | 5.3s | 5.3s | (unresolvable) | n/a |
| short | **cg_nosim** | **8.7s** | 7.7s | 1.08s | 14.6x |
| short | **cg_sim** | **18.8s** | 15.0s | 3.82s | 51.8x |

| Bench | Ir per body | Ir fixed | D refs per body | D1 miss rate | LL misses per body |
|---|---|---|---|---|---|
| long | 5,818,846,953 | 1,651,734,501 | 2,084,542,338 | 3.08% | 955 |
| short | 421,622,472 | 1,370,800,269 | 163,205,071 | 1.14% | 1,057 |

### Findings

1. **Determinism: Ir spread 0.00002% over 3 runs** (about 84 instructions of 421M). Wall clock per-pair spread in C was 0.89 to 1.55%.
2. **Fixed cost dominates short bodies:** 89% of cg_nosim time on the short bench is fixed. Shrinking inputs does not reduce counting cost. Untested remedy: start instrumentation after import (`callgrind --instr-atstart=no`).
3. **This target fits in LL cache.** LL misses are 0.00005% of data refs on the long bench; the graph fits the simulated 33 MiB LL. Memory layout effects at this size are visible only as D1 misses. Cachegrind's cache model is a simulation of the detected host configuration, not the hardware.
4. **Results identical across all four configurations** (fingerprints match).
5. **nulgrind short body is unresolvable** (70ms body under the 1x/2x subtraction). Discard that cell.

### Consequence

Counting runs on separate boxes in parallel with the referee, because counts are unaffected by contention and drift. It adds no referee queue time. cg_sim costs 2.5x cg_nosim and shows almost no LL signal on this target. Option for the user: count on every candidate, simulate cache only when the Ir delta and wall delta disagree.

> **Not what was built, and now retired.** The referee counted synchronously on its own box, which check 3 measured at 62.8% of all box seconds. Rather than move counting onto its own fan out, it was removed: the Ir delta and the wall delta disagreed by about 2.1x on every bitset patch, and cachegrind cost roughly 50,000x on the vectorized ones. Full working in `artifacts/instruction_counting.md`.
