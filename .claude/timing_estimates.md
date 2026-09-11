# Timing estimates for the width experiment

Written 2026-09-11. Every number here is either measured in Phase 1 or an assumption. The source column says which. Rescale everything the moment the pilot replaces the worker assumption.

## Definitions

| Term | Meaning |
|---|---|
| Worker | One AI agent. It reads the profile and all earlier results, then proposes one patch. |
| Referee | One program on its own sailbox. It applies a patch, runs the tests, times it against the current code, counts instructions, and returns accept or reject. |
| Round | A group of workers started at the same moment. Each sees all earlier rounds' results and none of the other patches in its own round. |
| Width | Workers per round. |
| Setting | One width, run for 32 rounds. |
| Target | One piece of networkx code plus the one benchmark that measures it. Only `nx.clustering` on the 1000 node directed graph is chosen so far. |
| Pilot | The first 20 rounds of the width 1 setting on target 1, watched by hand. Not a separate run. |

## Parallelization scheme

- One referee box per worker. Measurement B showed load in the same box shifts timings by 5.77 percent, so timing needs a dedicated box, and load in a neighbouring box cancels in back to back pairs, so many referees can run at once.
- Widths within a target run at the same time, except that widths 4 and 16 wait for the pilot's round 20 checkpoint.
- Targets run one after the other.
- Rounds are sequential by definition. Each round reads the previous round's results.

## Cost of one round for one worker

| Step | Seconds | Source |
|---|---|---|
| Worker think time | 600 | Assumption. Not measured. The pilot measures it. |
| Test suite, 4 cores with pytest xdist | 61 to 101 | Measured twice during intake. 101 used below. |
| Wall clock timing, 6 back to back comparisons on the large graph | 128 | Measured in C. |
| Instruction count | 24 | Measured in D. |
| Total | 853 | |

The three referee steps run one after the other on the worker's referee box. Tests first, and timing plus counting only if the tests pass. A round that fails tests costs 701 seconds, but the estimates below assume every patch passes, which is the worst case.

## Time per setting

32 rounds at 853 seconds is 27,296 seconds, which is 7.58 hours. Round 20 lands at 4.74 hours.

A round ends when its slowest worker's patch has been judged. The 853 figure assumes every worker takes exactly 600 seconds. At width 16 the slowest of 16 model calls is longer than a typical one, so the width 16 setting will run over 7.58 hours by however much the worst call exceeds the typical one. A hard cap on worker time bounds this.

## Chosen sequence

| Stage | Rounds | Hours | Running total |
|---|---|---|---|
| Target 1, width 1, pilot phase, watched | 20 | 4.74 | 4.74 |
| Target 1, widths 4 and 16 in parallel. Width 1 finishes its last 12 rounds alongside them. | 32 | 7.58 | 12.32 |
| Target 2, widths 1, 4 and 16 in parallel | 32 | 7.58 | 19.90 |

19.9 hours of machine time. The two decision points add human latency on top, which is not counted.

## Alternatives that were costed and not chosen

| Scheme | Hours | Why not |
|---|---|---|
| Everything in parallel, both targets, no pilot gate | 7.58 | A harness bug found at round 3 has already burned 21 workers and 21 referees across two targets. |
| Both targets sequential, widths in parallel, no pilot gate | 15.17 | Same problem for one target. |
| Pilot alone, then widths 4 and 16 after width 1 finishes all 32 rounds | 22.70 | Waiting for the last 12 rounds of width 1 buys nothing. The pilot's decision is made at round 20. |

## What the round 20 checkpoint decides

Widths 4 and 16 launch only if all four hold:

1. Zero harness errors. No crashed rounds, no referee timeouts, no lost boxes.
2. Worker seconds per round is measured and replaces the 600 second assumption.
3. Tokens per patch is measured, so the credit cost of the wider settings is known before committing to them.
4. Test pass rate and accept count are known, so it is clear whether agents produce anything.

Launching at round 20 requires that nothing changes after the pilot. Widths 4 and 16 must use the same prompt, model, referee threshold and target as width 1, or the three settings are not comparable. If the pilot shows a prompt bug or a wrong threshold, restart all three settings together.

## Further time savings not adopted

- Tests and instruction count on a second box, concurrent with timing. Tests plus counting take about 125 seconds and timing takes 128. Round drops from 853 to about 728 seconds, saving close to an hour per setting. Costs two boxes per worker instead of one. Not adopted, to keep the pipeline simple.
- A larger sailbox for the tests, if Sail offers one. Not checked.
- Faster model tier, capped output length, or a cached fixed prefix in the prompt. These cut the 600 second term directly, and that term is 70 percent of the round. Which of them apply depends on the worker model, which is not chosen yet.

## Sensitivity

Everything above scales one for one with worker think time. If the worker averages 3 minutes, a setting is about 3.6 hours. If it averages 20 minutes, about 12.7 hours.

## Concurrency check

Stage two runs 21 workers and 21 referee boxes at once on target 1. Confirm the Sail account allows that many concurrent boxes before relying on the 19.9 hour figure. Not yet checked.
