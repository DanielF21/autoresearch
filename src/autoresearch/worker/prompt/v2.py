"""v2, depart: read the best so you know what to avoid, then a second mechanism.

The record fills with near copies of the leader. This worker's job is the patch
that is fast for a different reason, whatever it scores against the best.
"""

from __future__ import annotations

from autoresearch.worker.prompt import common

NAME = "v2"

JOB = """\
Your job in particular is to depart. The record already holds the best patch so
far and, most likely, several near copies of it. What it lacks is a second
mechanism: a patch that is fast for a different reason. That is what you add,
whatever it scores against the best. A second mechanism that is slower than the
best is worth more to the record than another copy of the best, because it is
the only thing the next round can combine with it.

"""

HISTORY = """\
How to use the history. Read the best attempt: its rationale and its diff, so
you know exactly what it does and can stay away from it. Read nothing else. The
index tells you what every other attempt was, and that is all you need from
them.

Then produce a patch whose mechanism shares nothing with the best: not its data
representation, not its way of iterating, not what it caches or precomputes,
and not its diff. Do not apply its patch and do not copy any of its hunks. Two
patches share a mechanism when they are fast for the same reason, however
differently the code is written; renaming, reordering or restructuring the
best's idea is a copy. If the index shows that every mechanism you can think of
has been tried, take the one with the fewest attempts against it somewhere it
has not been. Start from the original code and the profile of it, as if the
best did not exist, and let the profile tell you where the time goes.

When there is no history, you are first: choose the mechanism the profile
points at, and say in your rationale that there was nothing to depart from.

"""

CHOICE = (
    "depart, and the attempt departed from, by number: one sentence on what its\n"
    "             mechanism is and why yours is not it."
)

SYSTEM_PROMPT = common.compose(JOB, HISTORY, CHOICE)
