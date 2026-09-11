# Sailbox purposes

| sailbox_name | purpose |
|---|---|
| phase0-20260911-100411 | Phase 0 environment characterization: cgroup quota, steal, SMT, PMU, clocksource, turbo control |
| cpuprobe-102603-0 | CPU homogeneity probe, batch 1 of 4 |
| cpuprobe-102603-1 | CPU homogeneity probe, batch 1 of 4 |
| cpuprobe-102603-2 | CPU homogeneity probe, batch 1 of 4 |
| cpuprobe-102603-3 | CPU homogeneity probe, batch 1 of 4 |
| cpuprobe-102653-0 | CPU homogeneity probe, batch 2 of 3, re-run after numpy 2.x probe fix |
| cpuprobe-102653-1 | CPU homogeneity probe, batch 2 of 3, re-run after numpy 2.x probe fix |
| cpuprobe-102653-2 | CPU homogeneity probe, batch 2 of 3, re-run after numpy 2.x probe fix |
| measure-a-20260911-103915 | Measurement A smoke test, k=3 m=3 sets=2, harness validation only |
| measure-a-20260911-104158 | Measurement A full run, k=30 m=11 sets=3, uncontended timing noise |
| measure-b-ref-20260911-111220 | Measurement B smoke test 1, referee |
| measure-b-load0-20260911-111220 | Measurement B smoke test 1, load generator |
| measure-b-load1-20260911-111220 | Measurement B smoke test 1, load generator |
| measure-b-ref-20260911-111410 | Measurement B smoke test 2, referee, verifying cross-box load actually starts |
| measure-b-load0-20260911-111410 | Measurement B smoke test 2, load generator |
| measure-b-load1-20260911-111410 | Measurement B smoke test 2, load generator |
| measure-b-ref-20260911-111520 | Measurement B first full attempt, referee. Crashed at 7 min on a pgrep parsing bug |
| measure-b-load0-20260911-111520 | Measurement B first full attempt, load generator |
| measure-b-load1-20260911-111520 | Measurement B first full attempt, load generator |
| measure-b-load2-20260911-111520 | Measurement B first full attempt, load generator |
| measure-b-ref-20260911-112121 | Measurement B smoke test 3, referee, verifying the load-off transition after the fix |
| measure-b-load0-20260911-112121 | Measurement B smoke test 3, load generator |
| measure-b-load1-20260911-112121 | Measurement B smoke test 3, load generator |
| measure-b-ref-20260911-112247 | Measurement B full run, referee. k=6 m=11 cycles=3, contended timing noise |
| measure-b-load0-20260911-112247 | Measurement B full run, load generator |
| measure-b-load1-20260911-112247 | Measurement B full run, load generator |
| measure-b-load2-20260911-112247 | Measurement B full run, load generator |
| intake-20260911-121912 | networkx intake, first attempt. Stopped at 3 min: no per-benchmark timeout, results saved only at the end |
| intake-20260911-122516 | networkx intake, second attempt. Ran clean but benchmark_algorithms.py failed to import on a sys.path bug |
| intake-20260911-122906 | networkx intake, final. 298 benchmarks enumerated, 286 timed. Chose the C and D target |
| measure-c0-20260911-124054 | Measurement C0 smoke test, null comparison of two unmodified checkouts |
| measure-c1-20260911-124054 | Measurement C1 smoke test, whitespace patch in cluster.py |
| measure-d-20260911-124054 | Measurement D smoke test, instruction counting cost |
| measure-c0-20260911-124254 | Measurement C0 full run, 12 quartets m=7, two unmodified checkouts |
| measure-c1-20260911-124254 | Measurement C1 full run, 12 quartets m=7, whitespace patch in cluster.py |
| measure-d-20260911-124254 | Measurement D full run, plain vs nulgrind vs cachegrind with and without cache simulation |
| profile-20260911-143647 | cProfile of the target, 3 runs per graph, dumps fetched for token counting |
| volprobe-20260911-161040-0 | Volume access check. Mounted a probe volume, wrote a file, ran git init on it. Size s |
| volprobe-20260911-161040-1 | Volume access check. Mounted the same volume after box 0 was terminated, read the file back. Size s |
