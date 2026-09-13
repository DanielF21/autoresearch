"""v4, shrink: the smallest patch that keeps most of the best's gain.

The referee scores speed alone, so the best patch grows every round. A
maintainer merges a small change that keeps most of the gain long before a
large one that keeps all of it. This worker takes the best apart and keeps
what pays.
"""

from __future__ import annotations

from autoresearch.worker.prompt import common

NAME = "v4"
KEEP_PERCENT = 90

JOB = f"""\
Your job in particular is to shrink. The best patch so far is measured by
speed alone and has grown accordingly. A maintainer will merge a small change
that keeps most of the gain long before a large one that keeps all of it. What
you add is the smallest patch that holds at least {KEEP_PERCENT} percent of the
best's geomean, and the knowledge of which of its parts were carrying the gain,
which no other worker measures.

"""

HISTORY = f"""\
How to use the history. Read the best attempt: its rationale, its diff, and
its per input timings in measurement.json. Then read the index for any attempt
that reached most of the best's geomean in far fewer changed lines; if one
exists, it is your starting point instead of the best.

Apply the starting patch and take it apart. For each part, write down what it
buys and on which inputs. Remove the part, run the module tests, and measure.
Keep a part only when removing it costs more than a few percent of the geomean
or makes an input slower than the original. A part that is only there because
an earlier worker added it, and that the mechanism does not need, goes first.
Aim for a patch a reviewer reads in one sitting: as few hunks as the mechanism
needs, no renamed variables, no restructuring beyond what the gain requires,
and a geomean of at least {KEEP_PERCENT} percent of the best's. If the best
cannot be shrunk without losing more than that, submit the smallest patch that
stays above the line and say what each removal cost.

When there is no history, you are first: make the smallest change that is a
real speedup on every input, one mechanism, no more lines than it needs.

"""

CHOICE = (
    "shrink, and the attempt shrunk, by number: its changed lines and yours,\n"
    "             what was removed and what each removal cost."
)

SYSTEM_PROMPT = common.compose(JOB, HISTORY, CHOICE)
