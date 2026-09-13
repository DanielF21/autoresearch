"""Can a reader pick the draft out from bodies the repository's contributors wrote?

One trial shows the draft among real holdout bodies in a seeded order, all
anonymised alike, and asks which one a model wrote. The score is the share of
valid trials in which the reader picked the draft. Chance is one over the number
of bodies shown; a draft that reads like the corpus should sit near it. Whether
this judge can tell anything at all is what calibration measures before its
score is trusted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from autoresearch.model.protocol import ChatModel, Message
from autoresearch.scribe.judges.common import anonymise, letters
from autoresearch.scribe.loop import Caps, LoopResult, run_agent
from autoresearch.scribe.tools import Roots, ToolContext, submit_tool
from autoresearch.types import Usage

TitledBody = tuple[str, str]
CANDIDATE = "candidate"


@dataclass(frozen=True)
class Trial:
    seed: int
    order: tuple[str, ...]
    candidate_letter: str
    picked_letter: str | None
    tells: tuple[str, ...]
    stop: str

    @property
    def valid(self) -> bool:
        return self.picked_letter is not None

    @property
    def picked_candidate(self) -> bool:
        return self.picked_letter == self.candidate_letter

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "order": list(self.order),
            "candidate_letter": self.candidate_letter,
            "picked_letter": self.picked_letter,
            "picked_candidate": self.picked_candidate,
            "tells": list(self.tells),
            "stop": self.stop,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trial:
        return cls(
            seed=int(d["seed"]),
            order=tuple(str(x) for x in d["order"]),
            candidate_letter=str(d["candidate_letter"]),
            picked_letter=None if d.get("picked_letter") is None else str(d["picked_letter"]),
            tells=tuple(str(x) for x in d.get("tells", [])),
            stop=str(d.get("stop", "")),
        )


@dataclass(frozen=True)
class Discrimination:
    trials: tuple[Trial, ...]
    shown: int
    usage: Usage

    @property
    def valid(self) -> int:
        return sum(1 for t in self.trials if t.valid)

    @property
    def picked(self) -> int:
        return sum(1 for t in self.trials if t.picked_candidate)

    @property
    def rate(self) -> float | None:
        return None if not self.valid else self.picked / self.valid

    @property
    def chance(self) -> float:
        return 1.0 / self.shown if self.shown else 0.0

    def tells(self) -> list[str]:
        return [tell for t in self.trials if t.picked_candidate for tell in t.tells]

    def to_dict(self) -> dict[str, Any]:
        return {
            "shown": self.shown,
            "chance": self.chance,
            "valid": self.valid,
            "picked": self.picked,
            "rate": self.rate,
            "trials": [t.to_dict() for t in self.trials],
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Discrimination:
        return cls(
            trials=tuple(Trial.from_dict(t) for t in d.get("trials", [])),
            shown=int(d["shown"]),
            usage=Usage.from_dict(d.get("usage", {})),
        )


def trial_message(bodies: list[tuple[str, TitledBody]], labels: list[str]) -> str:
    parts = [f"{len(bodies)} pull request descriptions follow. Exactly one was written by a model."]
    for label, (_, (title, body)) in zip(labels, bodies, strict=True):
        parts += ["", f"## Description {label}", f"Title: {anonymise(title)}", "", anonymise(body)]
    return "\n".join(parts)


def run_trial(
    model: ChatModel,
    prompt: str,
    candidate: TitledBody,
    decoys: list[TitledBody],
    seed: int,
    caps: Caps,
    *,
    cache_key: str,
) -> tuple[Trial, LoopResult]:
    items: list[tuple[str, TitledBody]] = [(CANDIDATE, candidate)]
    items += [(f"decoy{i}", d) for i, d in enumerate(decoys)]
    random.Random(seed).shuffle(items)
    labels = letters(len(items))

    def validate(args: dict[str, Any]) -> str | None:
        if args.get("letter") not in labels:
            return f"letter must be one of {labels}"
        tells = args.get("tells")
        if not isinstance(tells, list) or not all(isinstance(t, str) for t in tells):
            return "tells must be a list of strings"
        return None

    submit = submit_tool(
        "submit_pick",
        "The letter of the description a model wrote, and the tells.",
        {
            "type": "object",
            "properties": {
                "letter": {"type": "string", "enum": labels},
                "tells": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["letter", "tells"],
        },
        validate,
    )
    messages: list[Message] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": trial_message(items, labels)},
    ]
    result = run_agent(model, messages, (submit,), ToolContext(Roots()), caps, cache_key=cache_key)
    cand_letter = labels[[name for name, _ in items].index(CANDIDATE)]
    picked = None if result.submitted is None else str(result.submitted["letter"])
    tells = () if result.submitted is None else tuple(str(t) for t in result.submitted["tells"])
    trial = Trial(
        seed=seed,
        order=tuple(name for name, _ in items),
        candidate_letter=cand_letter,
        picked_letter=picked,
        tells=tells,
        stop=str(result.stop),
    )
    return trial, result


def discriminate(
    model: ChatModel,
    prompt: str,
    candidate: TitledBody,
    decoys: list[TitledBody],
    *,
    shuffles: int,
    seed: int,
    caps: Caps,
    cache_key: str,
) -> Discrimination:
    trials: list[Trial] = []
    usage = Usage()
    for s in range(shuffles):
        trial, result = run_trial(
            model, prompt, candidate, decoys, seed * 1000 + s, caps, cache_key=cache_key
        )
        trials.append(trial)
        usage = usage + result.usage
    return Discrimination(tuple(trials), len(decoys) + 1, usage)
