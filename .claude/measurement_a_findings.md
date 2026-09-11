# Measurement A: findings in plain English

**What the box did:** ran one small, pointless piece of arithmetic over and over, thousands of times, timing each run. Nothing to do with networkx. The only question was: if I ask this machine to do the exact same work twice, does it take the same amount of time?

That matters because the entire project rests on being able to tell "the patch made it faster" apart from "the machine was having a good moment." If the machine cannot give a consistent answer for identical work, it cannot be trusted to judge a patch.

## What we found

**1. The machine's speed changes over time, by nearly a tenth across twenty minutes.** The same fixed task ran progressively faster across the three checkpoints of this one session.

**Be careful about what that does and does not establish.** This is one box, one session, three checkpoints. Three points falling in order happens by chance one time in six. The honest claim is that **the box's speed is not stable over twenty minutes and can move by roughly nine percent.** Whether it reliably speeds up, or simply wanders and happened to wander downward this once, is not established. The cause is not visible from inside the VM.

The magnitude and the timescale are what matter for the project, and both are solid. The direction is not, and mostly does not need to be.

This is the dangerous one either way. If you time the original code now and the patched code twenty minutes later, the two can differ by several percent for reasons that have nothing to do with the patch. That is bigger than most of the real speedups you are hunting for. You would ship a patch that does nothing and have a benchmark showing it worked.

**2. Setting that aside, the machine is steady enough to use.** Identical runs close together cluster reasonably tightly. Not pristine, not disqualifying.

**3. But a single head to head comparison is too crude to trust.** Run the old code once and the new code once, back to back, and most of the gap between them is just the machine being the machine. You have to repeat the head to head several times before a genuine improvement separates from the fuzz. Roughly five repeats to reliably see a two percent win, twenty repeats to see a one percent win. That is the price, and it multiplies straight into how long the whole run takes.

**4. So: always time the before and after right next to each other, alternating, and never reuse an old measurement.** That was already the plan. It is now proven necessary rather than assumed.

**5. The canary paid for itself on its first outing.** Because a fixed, never changing reference task ran alongside everything else, the movement over time was obvious. Without it the numbers would have looked merely scattered, the conclusion would have been "noisy machine," and the referee would have been built broken.

## One correction worth flagging

The first pass at the data lumped "machine drifting" and "machine jittery" into a single number, and on that basis declared the platform unusable. They are different problems with different fixes. Separated properly, the platform is fine and the drift is something the experiment design already handles.

## Open question this raises

Is the drift directional (a warmup effect) or a random walk? It matters because a warmup effect has a cheap extra mitigation: burn the box in before measuring anything. A random walk has none, and adjacent pairing is the only defence. Measurement B runs canaries at the top of every cycle on a different box, so it collects the evidence to answer this for free.

---

Numbers and raw data: `artifacts/measurements.md`, `runs/measure-a/20260911-104158.jsonl`.
Box: `measure-a-20260911-104158`. 180 launches, ~19 minutes.
