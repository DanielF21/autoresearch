"""v1, the generalist: survey the whole history,
then choose combine, extend or depart from the current best patch."""

from __future__ import annotations

from autoresearch.worker.prompt import common

NAME = "v1"

JOB = """\
What makes an attempt worth its cost is what it adds to the record: a gain the
best patch so far does not have, a mechanism nobody has tried, a combination of
gains that so far exist only in separate attempts, or a clear answer to why an
idea does not work. Reproducing the best patch with small changes adds nothing,
whatever it scores.

"""

HISTORY = """\
How to use the history. It is evidence about what works on this code, gathered
by engineers who each saw less of it than you do now. It is not a list of
changes to reproduce, and the best number in it is not an instruction to start
from that patch. Before you change anything, survey it:
- The index gives every attempt's geomean, its worst input, its outcome, and
  the first line of its rationale, which names the mechanism. Read the whole
  index.
- Read the full rationale of the best attempt, of the best attempt that uses a
  different mechanism, of attempts that were slower on an input, and of the
  newest few. Rationales are short; diffs are not. Read a diff only when you
  intend to build on it.
- Every measurement.json has each input's own timing. Where the index shows a
  patch strong on one input and weak on another, that file says which.
When you are done you should be able to write down: the mechanisms tried, which
inputs each one helps, what has been slower and why, which gains exist in
separate attempts that no single patch combines, and one thing nobody has
tried. This takes a few commands. They are the best ones you will run.

Then choose one of three things to do, and say which in your rationale:
1. Combine. Gains that exist in different attempts and are not in the best
   patch. The value is the combination; check that the parts do not undo each
   other.
2. Extend. The best patch, or another, plus a new mechanism that changes where
   its time goes. Profile the patched tree first; the original's profile no
   longer describes it.
3. Depart. A mechanism nobody has tried, from the original code.
Two patches are different when they change the underlying mechanism, not when
the same mechanism is written differently. An earlier failure is evidence, not
a prohibition: an idea can fail on its implementation, on its cost on the small
inputs, or on a mistake, and any of those can be fixed. If you retry one, say
what is different.

"""

CHOICE = "combine, extend or depart, and the attempts built on, by number."

SYSTEM_PROMPT = common.compose(JOB, HISTORY, CHOICE)
