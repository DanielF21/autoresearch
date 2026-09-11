# Measurement B: findings in plain English

**What the boxes did:** four machines. One did the timing, the same pointless arithmetic as before. The other three were deliberately hammered with busywork. The timing machine ran the same task under three conditions, rotated back and forth so that nothing could be blamed on the order: quiet, busy with its own work, and quiet-itself-but-with-the-neighbours-hammered.

The question: can the machine that judges patches share hardware with the machines doing the work, or does it need to be left alone?

## What we found

**1. A machine cannot judge patches while also doing work. It runs about 6 percent slower.** When the timing machine had three other jobs of its own running, everything it timed came out roughly 6 percent slow, and the fixed reference task came out nearly 9 percent slow. This showed up in all three rounds, always in the same direction. It is not a fluke.

So the judging machine gets its own box. That costs about two dollars and the question is settled.

**2. Neighbouring machines being busy also slows things down, but only slightly.** Around 1.5 to 2.5 percent. Small, but it appeared in all three rounds in the same direction, so it is probably real rather than noise. The machines are not fully isolated from each other.

**3. That small neighbour effect probably does not matter, and here is why.** If everything slows down by the same couple of percent, and you are comparing the old code against the new code side by side under those same conditions, the slowdown hits both equally and cancels out. What would actually hurt is if the neighbours made results more erratic rather than just uniformly slower, and that effect was mild.

**This is the answer to the expensive question.** It means you can probably run many judging machines at once instead of one, which is what determines whether the whole project takes ten hours or one. Not proven, but the evidence points that way.

**4. The warning system we planned does not work.** The plan was to detect interference by reading a counter the operating system keeps for exactly this purpose. That counter was essentially zero in every condition, including the ones where we could plainly see the machine running slower. It was actually slightly higher when the machine was idle than when the neighbours were hammering it.

So the interference is real and the intended alarm is blind to it. Something else has to do that job, and the obvious candidate is the fixed reference task, which caught all of this without difficulty.

**5. The speed drift from last time did not repeat the same way.** The earlier machine got steadily faster over twenty minutes, by nearly a tenth. This one moved about two percent and then settled almost immediately. Both moved in the same direction, which mildly supports a warm-up explanation, but the size varies enormously between machines. You cannot predict how much any given machine will drift.

That reinforces the same conclusion as before: never compare two timings taken far apart, on any machine, ever.

## What this changes

- The judging machine gets a dedicated box. Settled.
- Many judging machines in parallel looks viable, which is the difference between a ten hour bottleneck and a one hour one.
- The planned interference alarm is replaced by the fixed reference task, which demonstrably works.
- Comparisons stay tightly paired in time, on one machine. Same conclusion as measurement A, now for a second independent reason.

---

Numbers and raw data: `artifacts/measurements.md`, `runs/measure-b/20260911-112247.jsonl`.
Boxes: `measure-b-ref-20260911-112247` plus 3 load boxes. 216 launches, ~25 minutes.
