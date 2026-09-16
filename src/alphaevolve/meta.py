"""Meta prompt evolution.

The paper: the prompts themselves are evolved, with instructions and context
suggested by the model in an additional prompt generation step and co-evolved
in a separate database analogous to the one for programs. It gives no
mechanics, and OpenEvolve does not implement it (its PromptConfig marks meta
prompting as not implemented). Everything below is this package's own choice:

- The population starts with one empty instruction, ``m0000``.
- In every batch, worker slot 0 makes one extra model call that reads the
  population with each instruction's record and writes one new instruction.
  It joins the population from the next batch, under the writer's attempt
  number.
- An instruction's fitness is the best fitness among the candidates that used
  it. One no candidate has used yet is weighted as the best fitness known, so
  it is tried.
- Each candidate draws one instruction with probability proportional to
  fitness, floored at 0.001, as OpenEvolve weights parents.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from alphaevolve.database import BASE_FITNESS, WEIGHT_FLOOR
from alphaevolve.records import Lineage, meta_id

INITIAL_ID = "m0000"
_TAG = re.compile(r"<instructions>\s*(.*?)\s*</instructions>", re.DOTALL)


@dataclass(frozen=True)
class MetaPrompt:
    id: str
    text: str
    uses: int
    best: float | None


def population(
    lineages: Mapping[int, Lineage], fitness: Mapping[int, float]
) -> tuple[MetaPrompt, ...]:
    """Every instruction written so far, with how the candidates using it scored."""
    texts: dict[str, str] = {INITIAL_ID: ""}
    for number in sorted(lineages):
        if lineages[number].meta_generated:
            texts[meta_id(number)] = lineages[number].meta_generated
    uses = dict.fromkeys(texts, 0)
    best: dict[str, float | None] = dict.fromkeys(texts)
    for number in sorted(lineages):
        used = lineages[number].meta
        if used not in texts:
            continue
        uses[used] += 1
        score = fitness.get(number, 0.0)
        previous = best[used]
        best[used] = score if previous is None else max(previous, score)
    return tuple(MetaPrompt(k, texts[k], uses[k], best[k]) for k in texts)


def choose(pool: Sequence[MetaPrompt], rng: random.Random) -> MetaPrompt:
    known = [p.best for p in pool if p.best is not None]
    optimistic = max(known) if known else BASE_FITNESS
    weights = [max(p.best if p.best is not None else optimistic, WEIGHT_FLOOR) for p in pool]
    return rng.choices(list(pool), weights=weights, k=1)[0]


def parse_generated(reply: str) -> str:
    """The new instruction in a meta reply, or empty when there is none."""
    m = _TAG.search(reply)
    return m.group(1).strip() if m else ""
