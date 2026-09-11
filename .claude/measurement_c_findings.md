# Measurement C: findings in plain English

## Why this matters to the project

The project ends in claims of the form "this patch made networkx X percent faster." The referee makes that call. It times the original code and the patched code and compares them. Measurements A and B measured noise on a synthetic arithmetic loop. The referee will time real networkx code. So the number every later claim depends on had not been measured yet: how small a speedup the referee can detect on real code.

## The question

If a patch changes nothing, how different do the before and after timings look anyway? Any real speedup has to be bigger than that difference to be believed.

## What we ran

Two machines, running at the same time. Each machine held two copies of networkx at the same commit.

**Machine C0.** Both copies unmodified. This checks whether having two copies at all introduces a difference.

**Machine C1.** One copy had a single blank line inserted in `networkx/algorithms/cluster.py`, above the first function. That shifts the line number of every function in the file, the way a real patch would. The computation is unchanged.

Both machines timed the same networkx function, `nx.clustering`, on two graphs with 1000 nodes each. The larger graph takes 1.39 seconds per call. The smaller takes 80 milliseconds. `cluster.py` accounts for 92 percent of the function's own compute time, so the blank line sits on the code that actually runs.

Each machine ran 24 comparisons per graph. A comparison is one timing of each copy, taken back to back. The order alternated so neither copy always went first. The results of every run were checked and were identical across both copies.

## What we found

**1. When nothing changed, the timings agreed.** The median ratio of original to modified time was 1.0012, 1.0009, 1.0033 and 0.9991 across the four combinations. A ratio of 1 means identical.

**2. Back to back pairing cancelled the drift.** During these runs, the fixed reference task moved by 6.6 and 8.7 percent from its fastest reading to its slowest. The comparisons still centred on 1. This is the design working as intended.

**3. The smallest believable speedup depends on how many comparisons you run.** Using all 48 null comparisons per graph, this is the speedup a patch must show to have less than a 5 percent chance of being noise:

| Comparisons | Large graph | Small graph |
|---|---|---|
| 1 | 1.65% | 3.77% |
| 3 | 1.12% | 1.67% |
| 5 | 0.72% | 1.53% |
| 10 | 0.56% | 1.18% |
| 24 | 0.41% | 0.79% |

**4. The small graph is noisier per comparison, but cheaper per unit of precision.** One comparison on the large graph takes 20.6 seconds. One on the small graph takes 1.6 seconds, because most of each run is spent in the function itself on the large graph. 24 comparisons on the small graph take 38 seconds and resolve 0.79 percent. 3 comparisons on the large graph take 62 seconds and resolve only 1.12 percent. Many cheap comparisons beat a few expensive ones.

**5. The blank line made no difference.** C1 matched C0 on both graphs. The concern that inserting a line would force Python to recompile the file during a timed run did not occur. Every measured run found its compiled copy already up to date, because recompilation happens during import, which is outside the timed section.

**6. One of the four null comparisons produced a false alarm.** On machine C0, small graph, the 90 percent interval around the median ratio was 1.0009 to 1.0124. It excluded 1, which would normally be read as a real difference. Nothing changed. With four such intervals, at least one false alarm is expected in about one run of every three. This is why the acceptance threshold has to come from the table above, not from asking whether an interval excludes 1.

**7. Real code was quieter than the synthetic loop.** Each comparison on the large graph varied by 0.89 percent. The synthetic loop in measurement A varied by 2.19 percent per comparison. These ran on different machines at different times, so the cause is not established.

**8. The drift direction repeated again.** All four machines that ran long enough ended faster than they started: 8.7, 2.2, 4.1 and 3.1 percent. If direction were random, four matching outcomes would happen by chance about one time in sixteen. But the C traces were not smooth. Six of eleven steps went up. The pattern looks like a general speed up with wandering on top. This is suggestive, not established.

## What this means for the project

The referee's cost and precision are now measured on real code rather than estimated.

**The cost per candidate on the large graph is 20.6 seconds per comparison.** Five comparisons cost 103 seconds and resolve 0.72 percent. The design document assumed 2 minutes per candidate. With the test suite added, at 61 to 101 seconds, a candidate costs roughly 3 to 3.5 minutes. For about 290 candidates that is 14 to 17 hours on one machine. Measurement B indicated that several referee machines can run at once, which divides that time.

**Running more comparisons on a shorter benchmark is the cheaper way to get precision.** There is a limit. The two graphs have different densities, so a patch to the inner loop can speed them up by different amounts. The referee should time the benchmark the claim will be made about.

**The plan's fixed floor of 2 percent is conservative for this target.** With 3 or more comparisons, the data supports claims below 2 percent. Whether to lower it is your decision.

**The acceptance rule is now concrete.** Accept a patch only if its median ratio across its comparisons exceeds the threshold in the table for that number of comparisons.

---

Numbers and raw data: `artifacts/measurements.md`, `runs/measure-cd/20260911-124254-c0.jsonl`, `-c1.jsonl`.
Boxes: `measure-c0-20260911-124254`, `measure-c1-20260911-124254`. 96 comparisons per box, about 10 minutes.
