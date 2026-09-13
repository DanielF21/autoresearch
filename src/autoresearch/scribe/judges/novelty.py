"""Is every claim in a draft supported, and does it say something the diff does not?

The judge splits the body into factual claims and labels where each comes from.
Code then holds the gates: no unsupported claim, every search log claim names an
attempt that exists, and, when the search had other attempts to learn from, at
least one claim comes from them. That last one is the "learn something about the
software" bar made checkable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoresearch.model.protocol import ChatModel, Message
from autoresearch.scribe.loop import Caps, run_agent
from autoresearch.scribe.tools import Roots, ToolContext, submit_tool
from autoresearch.types import Usage

SOURCES = ("from_diff", "from_measurement", "from_search_log", "unsupported")


@dataclass(frozen=True)
class Claim:
    text: str
    source: str
    attempt: int | None


@dataclass(frozen=True)
class Novelty:
    claims: tuple[Claim, ...]
    stop: str
    usage: Usage
    attempts: tuple[int, ...]
    require_search_log: bool

    @property
    def judged(self) -> bool:
        return bool(self.claims)

    @property
    def unsupported(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.source == "unsupported")

    @property
    def bad_refs(self) -> tuple[Claim, ...]:
        return tuple(
            c
            for c in self.claims
            if c.source == "from_search_log"
            and (c.attempt is None or c.attempt not in self.attempts)
        )

    @property
    def from_search_log(self) -> tuple[Claim, ...]:
        return tuple(
            c for c in self.claims if c.source == "from_search_log" and c not in self.bad_refs
        )

    @property
    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.judged:
            out.append(f"the claim judge returned nothing ({self.stop})")
        out += [f"unsupported: {c.text}" for c in self.unsupported]
        out += [f"cites an attempt that is not in the search: {c.text}" for c in self.bad_refs]
        if self.judged and self.require_search_log and not self.from_search_log:
            out.append("says nothing the search taught that the diff does not show")
        return tuple(out)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "claims": [
                {"text": c.text, "source": c.source, "attempt": c.attempt} for c in self.claims
            ],
            "stop": self.stop,
            "usage": self.usage.to_dict(),
            "attempts": list(self.attempts),
            "require_search_log": self.require_search_log,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Novelty:
        return cls(
            claims=tuple(
                Claim(
                    str(c["text"]),
                    str(c["source"]),
                    None if c.get("attempt") is None else int(c["attempt"]),
                )
                for c in d.get("claims", [])
            ),
            stop=str(d.get("stop", "")),
            usage=Usage.from_dict(d.get("usage", {})),
            attempts=tuple(int(a) for a in d.get("attempts", [])),
            require_search_log=bool(d.get("require_search_log", False)),
        )


def validate_claims(args: dict[str, Any]) -> str | None:
    claims = args.get("claims")
    if not isinstance(claims, list) or not claims:
        return "claims must be a non empty list"
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("text"), str):
            return "every claim needs text"
        if c.get("source") not in SOURCES:
            return f"source must be one of {SOURCES}"
        attempt = c.get("attempt")
        if attempt is not None and not isinstance(attempt, int):
            return "attempt must be an integer or null"
    return None


SUBMIT_CLAIMS = submit_tool(
    "submit_claims",
    "The description's factual claims, each labelled with its source.",
    {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source": {"type": "string", "enum": list(SOURCES)},
                        "attempt": {"type": ["integer", "null"]},
                    },
                    "required": ["text", "source"],
                },
            }
        },
        "required": ["claims"],
    },
    validate_claims,
)


def judge_novelty(
    model: ChatModel,
    prompt: str,
    title: str,
    body: str,
    dossier: str,
    *,
    attempts: tuple[int, ...],
    require_search_log: bool,
    caps: Caps,
    cache_key: str,
) -> Novelty:
    messages: list[Message] = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"{dossier}\n\n# The description\n\nTitle: {title}\n\n{body}\n",
        },
    ]
    result = run_agent(
        model, messages, (SUBMIT_CLAIMS,), ToolContext(Roots()), caps, cache_key=cache_key
    )
    claims: tuple[Claim, ...] = ()
    if result.submitted is not None:
        claims = tuple(
            Claim(str(c["text"]), str(c["source"]), c.get("attempt"))
            for c in result.submitted["claims"]
        )
    return Novelty(claims, str(result.stop), result.usage, attempts, require_search_log)
