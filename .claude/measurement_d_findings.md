# Measurement D: findings in plain English

## Why this matters to the project

Every candidate patch will get two measurements: a wall clock timing and an instruction count. An instruction count is the number of machine instructions the program executed. It is produced by a tool called cachegrind, which runs the program on a simulated processor. Because the processor is simulated, the count does not depend on how busy or how fast the real machine is. Measurements A, B and C showed the real machines drift and interfere with each other. An instruction count is immune to both.

The simulation is slow. So the open question was price.

## The question

How much time does instruction counting add to each candidate? And how much more does it cost to also simulate the processor's cache, which is the part that can see memory layout effects?

## What we ran

One machine. The same networkx function and the same two graphs as measurement C: 1.39 seconds per call on the large graph, 80 milliseconds on the small one.

Each graph was run four ways:

- **Plain.** No simulation. The reference.
- **Valgrind alone.** Valgrind is the framework cachegrind runs inside. Run with no tool attached, it shows the minimum cost of simulation.
- **Instruction counting.** Cachegrind counting instructions only.
- **Instruction counting with cache simulation.** Cachegrind also modelling the processor's cache and counting how often data was not found in it.

Each configuration ran twice: once calling the function one time, and once calling it two times. The difference between the two runs gives the cost of the function itself. The rest is fixed cost: starting Python, importing networkx, and building the graph.

## What we found

**1. Instruction counting costs 24 seconds per candidate on the large graph. Cache simulation raises that to 60 seconds.**

| Graph | Plain | Instruction count | With cache simulation |
|---|---|---|---|
| Large, 1.39s | 1.7s | 23.8s | 60.3s |
| Small, 80ms | 0.3s | 8.7s | 18.8s |

**2. Most of the cost on the small graph is fixed.** Under instruction counting, the small graph took 7.7 seconds of fixed cost and 1.1 seconds for the function itself. On the large graph it was 9.0 seconds fixed and 14.8 seconds for the function. The function ran 10.5 times slower than normal on the large graph and 14.6 times slower on the small one. With cache simulation, the slowdown was 28 and 52 times.

**3. The count is exactly repeatable.** Three identical runs differed by 0.00002 percent, about 84 instructions out of 421 million. Wall clock comparisons in measurement C varied by 0.89 to 1.55 percent. The instruction count is roughly fifty thousand times more precise.

**4. For this target, the graph fits in the cache.** The cache simulation counts two kinds of miss. A first level miss means data was not in the processor's smallest, fastest cache. A last level miss means data had to come from main memory. On the large graph, 3.08 percent of data reads missed the first level cache. Only 955 reads per call missed the last level cache, out of 2.08 billion. The graph is small enough to sit entirely in the 33 megabyte cache. At this size, the memory layout effects you raised earlier can only show up in the first level cache.

**5. Simulation did not change the answers.** The function's results were identical in all four configurations.

**6. One number is meaningless.** Valgrind alone showed a function cost of zero on the small graph. The function takes 70 milliseconds, which is too small for the one call versus two calls subtraction to resolve. It has no effect on the other results.

## What this means for the project

**Instruction counting is affordable on every candidate.** On the large graph, a candidate already costs about 103 seconds of wall clock timing for five comparisons, plus 61 to 101 seconds of tests. Instruction counting adds 24 seconds. More importantly, it does not need to run on the referee's machine. The count is unaffected by contention or drift, so it can run on a separate machine at the same time as the timing, and it adds nothing to the referee's queue.

**The two measurements answer different questions.** The instruction count answers whether the patch changed how much work the program does, down to a hundredth of a percent from a single run. The wall clock answers whether the patch made the program faster. A patch can change one without the other, and the disagreement is informative.

**Shrinking the input will not make counting cheaper.** On the small graph, 89 percent of the counting time is fixed cost, spent before the function runs. The option that could cut it is starting the count only after networkx has been imported. That was not tested.

**Cache simulation is expensive, and on this target it shows little.** It costs 2.5 times as much as plain counting, and the last level cache signal is essentially zero because the graph fits. One option is to count instructions on every candidate and run cache simulation only when the instruction count and the wall clock disagree. That is where a memory layout effect would show up. This is your decision.

---

Numbers and raw data: `artifacts/measurements.md`, `runs/measure-cd/20260911-124254-d.jsonl`.
Box: `measure-d-20260911-124254`. About 6 minutes.
