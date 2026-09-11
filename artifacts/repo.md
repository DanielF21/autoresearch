# networkx

Pinned commit: `c94928ed9489` (main, 2026-09-11). Raw intake data: `runs/intake/20260911-122906.jsonl`.

## Build

```bash
git clone --depth 50 https://github.com/networkx/networkx && cd networkx
git checkout c94928ed9489
pip install -e . numpy scipy pandas
```

Measured on a size `m` sailbox: clone 1.0s, install 3.2s, import 0.8s. Pure Python, no compilation step. This does not generalize to repositories with C extensions.

`pandas` is required only by the Drug Interaction network benchmarks. Without it, 9 of them error.

## Test (the correctness gate)

```bash
pip install pytest pytest-xdist
python -m pytest networkx -q -n 4 -p no:cacheprovider
```

9028 passed, 61 skipped, 741 expected failures. **61s and 101s** on two runs across 4 cores. The referee pays this on every candidate.

## One benchmark, without ASV

Do not import `benchmarks/benchmarks/benchmark_algorithms.py` per timing. It builds every graph in the suite at import time and takes **29 seconds** to load. Build only the target graph with the suite's own constructor and seed:

```python
import networkx as nx
G = nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)   # DirectedAlgorithmBenchmarks, "Erdos Renyi (1000, 0.05)"
nx.clustering(G)                                                # the body of time_clustering
```

Implemented in `inbox/null_worker.py`. The benchmark classes use ASV conventions: `params`, `param_names`, `setup(self, *params)`, `time_*(self, *params)`.

## Chosen referee target

`DirectedAlgorithmBenchmarks.time_clustering`, hot path `networkx/algorithms/cluster.py` (`clustering` at line 385, `_directed_triangles_and_degree_iter` at line 161).

| Size | Body | Setup | Role |
|---|---|---|---|
| Erdos Renyi (1000, 0.05) | 1636ms | 0.0ms | Long anchor |
| Erdos Renyi (1000, 0.01) | 89ms | 0.0ms | Short anchor |

Chosen because the body is in the 50ms to 2s band, setup is zero, the graph is seeded so every run measures the same input, and the same function is benchmarked at 17 graph shapes, which gives a regression guard set for free.

## Benchmark inventory

298 benchmark instances across 13 files. 286 timed, 1 exceeded the 3 second cap, 11 errored, 0 unreached.

| Did not run | Count | Cause |
|---|---|---|
| `DirectedAlgorithmBenchmarks.time_clustering(Complete (1000))` | 1 | Exceeded 3s cap |
| Drug Interaction network benchmarks | 9 | `pandas` missing from the image |
| `HitsAlgorithmBenchmarks.time_hits(grid(30, 30))` | 1 | Algorithm fails to converge on its own benchmark input |
| `KFactorBenchmarks.time_k_factor` | 1 | `setup()` signature does not match its params |

## Benchmarks in the usable band

Body inside 50ms to 2s, setup under 100ms, ranked by body duration. An Amdahl ceiling belongs to a function inside a benchmark, not to the benchmark itself, so ceilings appear only in the target profile below.

| Benchmark | Body |
|---|---|
| `DirectedAlgorithmBenchmarks.time_clustering(Erdos Renyi (1000, 0.05))` | 1636ms |
| `DirectedAlgorithmBenchmarks.time_clustering(Complete (100))` | 595ms |
| `ReachabilityBenchmark.time_is_reachable(Tournament (1000, seed=42))` | 385ms |
| `DirectedAlgorithmBenchmarks.time_clustering(Erdos Renyi (10000, 0.0005))` | 326ms |
| `DirectedAlgorithmBenchmarks.time_clustering(Erdos Renyi (100, 0.5))` | 196ms |
| `UndirectedAlgorithmBenchmarks.time_greedy_modularity_communities(...)` | 89ms |
| `DirectedAlgorithmBenchmarks.time_clustering(Erdos Renyi (1000, 0.01))` | 89ms |
| `DirectedAlgorithmBenchmarks.time_kosaraju_scc(Complete (1000))` | 88ms |

## Target profile

cProfile of `nx.clustering(G)`, 3 runs per graph, size `m` sailbox `profile-20260911-143647`. Raw: `runs/profile/20260911-143647/` (binary `.prof` dumps, text renderings, `summary.json`, `tokens.json`).

| Graph | Unprofiled | Profiled | cProfile overhead | Functions recorded |
|---|---|---|---|---|
| long, p=0.05 | 1494ms | 1653 to 1710ms | 1.13x | 16 |
| short, p=0.01 | 82ms | 102 to 104ms | 1.25x | 16 |

### Hot spots and Amdahl ceilings

Self share is the fraction of total time spent in the function's own code. The self ceiling is the speedup if that time went to zero: `1 / (1 - self share)`.

| Function | Kind | Calls, long | Self share, long (3 runs) | Self ceiling, long | Self share, short | Self ceiling, short |
|---|---|---|---|---|---|---|
| `cluster.py:160 _directed_triangles_and_degree_iter` | networkx | 1,001 | 81.8% (81.8 to 82.1) | 5.48x | 84.0% | 6.24x |
| `cluster.py:180 <genexpr>`, the `sum(1 for k in chain(...))` | networkx | 1,083,208 | 10.3% | 1.12x | 8.2% | 1.09x |
| `builtins.sum` | builtin | 99,508 | 7.8% | 1.08x (1.22x with its generator) | 6.6% | 1.07x (1.17x) |
| everything else, 13 functions | | | under 0.2% | | under 1.5% | |

Self time by kind, long: networkx 92.2%, builtins 7.8%. No numpy or scipy code runs. All of it is patchable Python in one file.

### Token counts

| Rendering | Bytes | o200k_base | DeepSeek-V3 | GLM-4.5 |
|---|---|---|---|---|
| Flat profile, by self time | 1,795 | 670 | 609 | 675 |
| Callers | 3,147 | 720 | 711 | 715 |
| Callees | 3,666 | 693 | 689 | 689 |
| All of `cluster.py` | 24,837 | 6,861 | 7,046 | 6,827 |
| Hot function, lines 160 to 193 | 1,134 | 298 | 324 | 296 |

Paths are repo relative. Absolute paths add 80 to 125 tokens per rendering. DeepSeek-V3 and GLM-4.5 are proxies for the newer versions in the design doc's cost table; the exact tokenizers for those are unknown.

### Profile findings

1. **The profile fits in a worker's context with room to spare.** Profile plus call graph plus the whole source file is under 9,000 tokens. The design doc made reading the profile through an RLM conditional on this count. On this target it is not needed.
2. **A function level profile gives one answer.** 82 percent of self time sits inside one 33 line function, and the profile cannot say which lines. Line level attribution was not measured.
3. **Shares are stable.** The top function's share ranged 81.8 to 82.1 percent across 3 runs.
4. **cProfile distorts little here** (1.13x and 1.25x), because the work happens inside a small number of function calls.
5. **Observation, not measured:** lines 178 and 179 build `set(G._pred[j])` and `set(G._succ[j])` for every neighbour `j` of every node `i`. Each node's neighbour sets are therefore rebuilt once per neighbour that points to or from it, rather than once in total. A patch that builds them once is a candidate for the known real speedup needed to validate the referee.

## Intake findings

1. **The undirected benchmarks build unseeded random graphs.** `UndirectedAlgorithmBenchmarks._graphs` calls `nx.erdos_renyi_graph(nodes, 0.1)` with no seed, so every process benchmarks a different graph. The directed ones pass `seed=42`. Any referee target from the undirected class needs a seed added in the worker, or its timings include graph variation.
2. **Graphs are built at import, not in `setup()`.** Loading `benchmark_algorithms.py` costs 29 seconds. Discovery through ASV pays this once; a referee that re-imports per timing would pay it every time.
3. **ASV is bypassed.** It builds per-commit environments and tunes repetition counts itself, neither of which the referee can hold fixed across two arms of a comparison.
4. **`benchmark_algorithms.py` imports its own package as `benchmarks.utils`.** The parent directory of the package must be on the path.
