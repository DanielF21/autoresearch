# One benchmark input was not enough: what the patches actually did

Established 2026-09-12, from the 24 patches recorded in `runs/t1_w4` and `runs/t1_w4b`.

The referee timed one input: `nx.clustering(G)` on
`erdos_renyi_graph(1000, 0.05, seed=42, directed=True)`. `TargetSpec`'s docstring said so,
"One piece of code plus the one benchmark that measures it." Four of the seven best
patches are up to 4x **slower** than stock on sparse graphs, and nothing in the harness
could see it. This file is the evidence, and the reason the referee now times five inputs
and grades on their geometric mean.

Every measurement here is local, on a networkx checkout at base sha
`c94928ed94899033126c9d47f797a1f698584b20`, comparing a patched tree against stock on the
same machine in the same process pattern. Absolute times are not comparable to a sailbox.
Ratios between two implementations on one machine are the finding.

---

## 1. Four of the seven best patches regress

Each patch applied to a clean checkout, timed at five densities at n=1000, min of three
runs. `bench` is the number the referee recorded.

```
           patch    bench    d=0.5    d=2.0    d=5.0   d=20.0   d=50.0    worst
     t1_w4b/0001    15.0x    1.87x    2.27x    3.63x    8.08x   18.26x    1.87x   safe
      t1_w4/0002    19.5x    1.81x    2.68x    4.36x   10.82x   23.50x    1.81x   safe
     t1_w4b/0004    19.9x    1.46x    2.48x    4.23x   10.51x   23.84x    1.46x   safe
     t1_w4b/0008    43.4x    0.30x    1.10x    3.98x   27.93x  117.68x    0.30x   REGRESSES
      t1_w4/0007    46.3x    0.35x    1.39x    5.03x   32.95x  136.99x    0.35x   REGRESSES
     t1_w4b/0010    69.1x    0.22x    0.81x    3.02x   26.17x  139.69x    0.22x   REGRESSES
      t1_w4/0012    78.8x    0.25x    1.02x    4.22x   36.46x  156.67x    0.25x   REGRESSES
```

All seven return correct results on every graph. The split is clean and it lands at 20x:
everything at or below 20x is safe, everything at or above 43x regresses.

**Generality fell as speed rose.** That is not luck, it is the gradient. General
optimisations get exhausted first, so the longer the search runs the more input specific
the leader becomes.

## 2. The rule that separates them is readable off the diff

The safe patches keep the cost model. The bitset patches still scale with degree; they do
the same work with a cheaper constant, so there is no input where they can lose.

The unsafe patches change what the cost depends on. A dense `n^3` einsum scales with node
count and ignores edges entirely, so it is a trade, and a trade always has a losing side.

Sweeping density at n=1000 for `t1_w4b/0010` puts the crossover at average degree 2.2:

```
       p  avg deg     stock   patched   speedup
  0.0005      0.5   0.0014s   0.0055s     0.25x  loses
   0.001      1.0   0.0024s   0.0054s     0.44x  loses
  0.0015      1.5   0.0036s   0.0057s     0.64x  loses
   0.002      2.0   0.0051s   0.0053s     0.96x  loses
   0.003      3.0   0.0087s   0.0057s     1.53x
   0.005      5.0   0.0177s   0.0058s     3.07x
    0.01     10.0   0.0487s   0.0057s     8.59x
    0.02     20.0   0.1565s   0.0059s    26.61x
    0.05     49.8   0.8760s   0.0067s   131.72x  <-- the benchmark
```

The patched column is flat at about 5.5ms regardless of density. That is the floor of the
dense path, and on a graph whose whole runtime is 1.4ms it is pure overhead. The benchmark
sits at average degree 50, twenty times past the crossover, so the entire losing region is
off the map.

## 3. The patch's own guard is correct, and about the wrong thing

`t1_w4b/0010` guards its fast path with `G.number_of_nodes() <= 1400`. That bound is
derived and it is right: `W` entries are 0, 1 or 2, so the largest triangle sum is `8n^2`,
float32 represents integers exactly to `2^24 = 16,777,216`, and `8n^2 <= 2^24` gives
`n <= 1448`. The agent did that arithmetic properly and rounded down.

It is a **correctness** bound. The crossover is about **density**, and node count and
density are independent, so no node count guard can express it. The agent did the maths it
was asked for. Nobody asked for the other one.

## 4. It is not a correctness bug

Eleven shapes the benchmark never produces, checked against stock:

```
  OK   integer labels, small                          10.84x
  OK   string labels (non dense branch)               16.70x
  OK   with self loops                                13.05x
  OK   scale free, barabasi albert directed            0.53x   regression
  OK   small world, directed watts strogatz            8.85x
  OK   complete graph n=200                         1613.06x
  OK   empty graph, no nodes                           3.06x
  OK   one node, no edges                              0.01x
  OK   no edges, 500 nodes                             0.17x   regression
  OK   n=1500, above the 1400 guard (falls back)       0.99x
  OK   nodes= subset argument (falls back)             0.96x
```

Correct in 11 of 11. The scale free case matters most: `gn_graph` is a growing network with
average degree about 1, which is the shape of trees, DAGs, citation graphs and dependency
graphs. Those are common, and the patch makes them twice as slow.

## 5. Nothing in the harness could have caught it

Three checks existed and none of them can see a performance regression.

- **9090 tests** check correctness. A patch that is 2x slower passes every one.
- **The referee** timed one graph. `graph: str` was a single string, eval'd once.
- **`run_benchmark`**, the agent's own tool, timed that same single graph. The agent had no
  instrument that could have shown it either, so this is not a case of ignoring evidence.

The loop did exactly what it was built to do.

## 6. The fix is not a filter: a better patch was being hidden

One extra condition on the same fast path, on density instead of node count:

```
                    d=0.5    d=2.0    d=5.0   d=20.0   d=50.0    worst
0010 as is          0.26x    0.95x    3.19x   26.87x  130.10x    0.26x
0010 + density      0.95x    0.97x    2.73x   24.97x  128.57x    0.95x
```

The regression goes from 0.26x to 0.95x and the benchmark keeps 128x. Correct in both
cases. Two honest limits: 0.95x is near neutral, not provably free, and the threshold used
(`edges > 3 * nodes`) is a guess just above the measured 2.2 crossover rather than a
derived bound. A real patch would derive it.

**No agent wrote that line because nothing scored it.** The referee was not merely failing
to reject a bad patch, it was hiding a better one.

## 7. Why the geometric mean, and why a gate as well

The recorded speedup is now the geometric mean across the five inputs. It penalises
imbalance by construction: `GM = AM * exp(-sigma^2 / 2)` in log space, so spread costs a
squared deviation term. The discount tracks spikiness exactly:

```
     patch   geomean     arith  log var   AM->GM discount
    bitset      5.21      8.50     1.00              1.6x
      gemm      4.56     33.98     5.42              7.5x
   guarded      6.05     31.64     3.75              5.2x
```

A uniform doubling multiplies the geomean by 2; a doubling on one input of five multiplies
it by 1.149. Breadth is worth exactly five times depth, and a 100x spike with four neutral
inputs ties a flat 2.512x. On the real patches the geomean alone already ranks the guarded
version above the simple bitset one above the regressing original, with no gate involved.

The separate no regression gate exists for one narrow reason: a geomean prices a regression
as a *finite* penalty, so a large enough spike can outrun it. `[10000, 1, 1, 1, 0.5]` scores
5.49 and would outrank the bitset patch at 5.21 while being half speed on one input. The
gate makes "slower nowhere" a hard constraint rather than a purchasable one.

## 8. One noise floor does not cover every input

`noise_floor = 1.0106` came from measurement C, 48 pairs on the dense graph. A faster call
gives a noisier ratio. Base against base, six pairs of seven repeats, 40 trials, the same
method measurement C used:

```
                      input      call   median  p99 ratio    worst
  mid        n=1000 p=0.01    0.0504s   1.0004     1.0127   1.0141
  sparse     n=1000 p=0.002   0.0052s   1.0004     1.0201   1.0313
 very sparse n=1000 p=.0005   0.0014s   0.9996     1.0210   1.0245
  scale free  gn_graph(800)   0.0019s   1.0010     1.0135   1.0160
```

Medians sit on 1.0, so there is no bias, but a 1ms call scatters roughly twice as wide as
the 1.4s benchmark. Every input therefore carries its own floor, and a regression is
`speedup < 1 / noise_floor` so the test is symmetric.

### 8a. The floors actually used, measured on a referee box

Those figures were a laptop's and only the reason per input floors exist. The floors in
`configs/t1_w4c.toml` were measured on a referee box on 2026-09-12: two worktrees of the
base commit, nothing patched, timed against each other through `Referee.null_pairs`, which
is the referee's own `_time_pairs` with the same pairing plan, hash seeds, core pin and
seven repeats per launch. 42 pairs per input, floor at the 99th percentile of the median of
six, bootstrapped from the pairs measured. Record:
`artifacts/calibration/20260912-161509.jsonl`.

```
        input     call      med       sd      p05      p95      min      max    floor
   er1000_005   1.4688s   1.0015   0.0079   0.9902   1.0131   0.9820   1.0263   1.0082
   er1000_001   0.0834s   1.0035   0.0106   0.9918   1.0214   0.9838   1.0376   1.0151
  er1000_0002   0.0094s   1.0059   0.0194   0.9821   1.0390   0.9470   1.0461   1.0280
        gn800   0.0038s   1.0027   0.0214   0.9799   1.0332   0.9117   1.0438   1.0218
   er1600_002   0.9613s   1.0027   0.0569   0.9820   1.0957   0.7332   1.2003   1.0193
```

Three things the box said that the laptop could not.

**The procedure agrees with measurement C.** `er1000_005` came back at 1.0082 against
measurement C's 1.0106 from 48 pairs. 0.24 percentage points apart by an independent route,
which is what licenses the other four numbers. The config keeps 1.0106: it is the stricter
of the two, and it keeps that column comparable with `t1_w4` and `t1_w4b`.

**Spread tracks call length, as predicted, over four orders of magnitude.** Standard
deviation of the null ratio runs 0.0079 at a 1.5 second call and 0.0214 at a 4 millisecond
one, 2.7x wider. A single floor would have been either too loose for `er1000_005` or too
tight for `gn800`. Every measured floor is tighter than the laptop provisional it replaced.

**`er1600_002` has a tail nothing else has, and the guard misses most of it.** Its call is
0.96 seconds, nearly as long as `er1000_005`, yet its standard deviation is 7x larger. The
steal guard flagged two pairs; two more sat at 0.73x and 1.20x and were recorded clean. The
floor is unmoved because it is a median of six and two outliers in 42 cannot shift that,
which is the reason the statistic is a median. It does mean this input's individual pairs
are not trustworthy, only its median.

**The loose end.** `er1000_0002` floors at 1.0280, so a regression there is only visible
below 0.973x. A patch 2% slower on that input is inside its noise and will be recorded as
no change. The gate catches the 0.22x class of regression this whole change exists for; it
does not catch a marginal one on the two cheapest inputs. Closing that needs more pairs on
the cheap inputs, which are cheap to run, and it has not been done.

## 9. What this cost

Fixed overhead of a launch, measured: interpreter start plus networkx import 0.07 to
0.12s, graph build 0.03 to 0.08s, so roughly 0.3s, matching `LAUNCH_FIXED_S` in
`profile_run.py`. (Since the harness stopped assuming networkx that constant is
`LEGACY_LAUNCH_FIXED_S`, used only for records written before the guest began reporting
its own process age per launch, which `PairTiming.base_fixed_s` now carries.) Four extra inputs add about 40s to a 134s referee attempt, which is
about 5% of a round's wall clock. The sparse inputs are nearly free because their calls
are milliseconds; the dense one still dominates.

## 10. What is still not measured

The five inputs are all synthetic. networkx's own asv suite parameterises over
`ER(100, 0.1 / 0.5 / 0.9)` plus a fetched drug interaction network, and all three of those
densities sit above our crossover, so the repo's own suite would probably have missed this
too. Real world topologies are the strongest mergeability signal available and they are
not represented here. Doing that needs a graph vendored into the box image, which is a
separate decision.
