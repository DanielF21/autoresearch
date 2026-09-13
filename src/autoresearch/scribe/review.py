"""The mergeability reviewer, and the comparison that picks among mergeable patches.

The reviewer reads the base and patched trees and states, with line citations,
what each patch does to the cost of the call and which inputs it could slow down.
In ``diff_only`` mode it sees no measurements, no rationale and no transcripts, so
a verdict comes from the code alone. Code checks that every citation points at
lines that exist; a verdict whose evidence does not check out counts as no verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoresearch.model.protocol import ChatModel, Message
from autoresearch.scribe.facts import Facts
from autoresearch.scribe.loop import Caps, LoopResult, run_agent
from autoresearch.scribe.method import WriterMethod
from autoresearch.scribe.render import facts_block, inputs_table, tests_line
from autoresearch.scribe.runread import Candidate
from autoresearch.scribe.tools import READ_TOOLS, Roots, ToolContext, ToolError, submit_tool

VERDICTS = ("mergeable", "needs_changes", "not_mergeable")
CONCERN_KINDS = (
    "perf_regression",
    "correctness",
    "api",
    "dependency",
    "readability",
    "scope",
    "style",
)


@dataclass(frozen=True)
class Evidence:
    path: str
    start: int
    end: int


@dataclass(frozen=True)
class Concern:
    kind: str
    claim: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Verdict:
    verdict: str
    cost_before: str
    cost_after: str
    cost_changed: bool
    inputs_at_risk: tuple[str, ...]
    concerns: tuple[Concern, ...]
    reasons: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "cost_model": {
                "before": self.cost_before,
                "after": self.cost_after,
                "changed": self.cost_changed,
            },
            "inputs_at_risk": list(self.inputs_at_risk),
            "concerns": [
                {
                    "kind": c.kind,
                    "claim": c.claim,
                    "evidence": [
                        {"path": e.path, "start": e.start, "end": e.end} for e in c.evidence
                    ],
                }
                for c in self.concerns
            ],
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verdict:
        cost = d.get("cost_model") or {}
        return cls(
            verdict=str(d["verdict"]),
            cost_before=str(cost.get("before", "")),
            cost_after=str(cost.get("after", "")),
            cost_changed=bool(cost.get("changed", False)),
            inputs_at_risk=tuple(str(x) for x in d.get("inputs_at_risk", [])),
            concerns=tuple(
                Concern(
                    kind=str(c.get("kind", "")),
                    claim=str(c.get("claim", "")),
                    evidence=tuple(
                        Evidence(
                            str(e.get("path", "")), int(e.get("start", 0)), int(e.get("end", 0))
                        )
                        for e in c.get("evidence", [])
                    ),
                )
                for c in d.get("concerns", [])
            ),
            reasons=str(d.get("reasons", "")),
        )


_EVIDENCE = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "root:relative/path"},
        "start": {"type": "integer"},
        "end": {"type": "integer"},
    },
    "required": ["path", "start", "end"],
}

VERDICT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "cost_model": {
            "type": "object",
            "properties": {
                "before": {"type": "string"},
                "after": {"type": "string"},
                "changed": {"type": "boolean"},
            },
            "required": ["before", "after", "changed"],
        },
        "inputs_at_risk": {"type": "array", "items": {"type": "string"}},
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(CONCERN_KINDS)},
                    "claim": {"type": "string"},
                    "evidence": {"type": "array", "items": _EVIDENCE},
                },
                "required": ["kind", "claim", "evidence"],
            },
        },
        "reasons": {"type": "string"},
    },
    "required": ["verdict", "cost_model", "inputs_at_risk", "concerns", "reasons"],
}


def validate_verdict(args: dict[str, Any]) -> str | None:
    if args.get("verdict") not in VERDICTS:
        return f"verdict must be one of {VERDICTS}"
    cost = args.get("cost_model")
    if not isinstance(cost, dict) or not all(k in cost for k in ("before", "after", "changed")):
        return "cost_model needs before, after and changed"
    if not isinstance(args.get("reasons"), str) or not args["reasons"].strip():
        return "reasons must be a non empty string"
    if not isinstance(args.get("inputs_at_risk"), list):
        return "inputs_at_risk must be a list"
    concerns = args.get("concerns")
    if not isinstance(concerns, list):
        return "concerns must be a list"
    for c in concerns:
        if not isinstance(c, dict) or c.get("kind") not in CONCERN_KINDS:
            return f"every concern needs a kind from {CONCERN_KINDS}"
        ev = c.get("evidence")
        if not isinstance(ev, list) or not ev:
            return "every concern needs at least one evidence citation"
        for e in ev:
            if not isinstance(e, dict) or not isinstance(e.get("path"), str):
                return "evidence items need path, start and end"
            if not isinstance(e.get("start"), int) or not isinstance(e.get("end"), int):
                return "evidence start and end must be integers"
    if args["verdict"] != "mergeable" and not concerns:
        return "a verdict other than mergeable needs at least one concern"
    return None


def check_evidence(verdict: Verdict, roots: Roots) -> tuple[str, ...]:
    """Every citation must name lines that exist in a directory root."""
    problems: list[str] = []
    for c in verdict.concerns:
        for e in c.evidence:
            try:
                root, rel = roots.split(e.path)
                if root not in roots.dirs:
                    raise ToolError(f"{root} is not a source tree")
                path = roots.real(root, rel)
                n = len(path.read_text(errors="replace").splitlines())
            except (ToolError, OSError) as err:
                problems.append(f"{e.path}: {err}")
                continue
            if not 1 <= e.start <= e.end <= n:
                problems.append(f"{e.path}: lines {e.start} to {e.end} are outside 1 to {n}")
    return tuple(problems)


def review_message(c: Candidate, facts: Facts, mode: str, roots: Roots) -> str:
    parts = [
        f"Review attempt {c.dirname}. Roots you can read: {', '.join(roots.names())}.",
        "base: is the repository at the base commit. patched: is the same commit with the patch applied.",
        "",
        "## Patch",
        "```diff",
        (c.patch or "").rstrip(),
        "```",
        "",
        "## Facts computed from the diff and the syntax trees",
        facts_block(facts),
    ]
    if mode == "full":
        parts += [
            "",
            "## Measurements",
            inputs_table(c),
            tests_line(c),
            "",
            "## The proposing agent's rationale",
            c.rationale.strip() or "(none)",
        ]
    return "\n".join(parts)


@dataclass(frozen=True)
class ReviewOutcome:
    verdict: Verdict | None
    evidence_problems: tuple[str, ...]
    loop: LoopResult

    @property
    def valid(self) -> bool:
        return self.verdict is not None and not self.evidence_problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": None if self.verdict is None else self.verdict.to_dict(),
            "valid": self.valid,
            "evidence_problems": list(self.evidence_problems),
            "stop": str(self.loop.stop),
            "turns": self.loop.turns,
            "error": self.loop.error,
        }


SUBMIT_VERDICT = submit_tool(
    "submit_verdict",
    "Submit the review. Every concern cites lines as root:path with start and end.",
    VERDICT_PARAMETERS,
    validate_verdict,
)


def review_candidate(
    model: ChatModel,
    method: WriterMethod,
    c: Candidate,
    facts: Facts,
    ctx: ToolContext,
    caps: Caps,
    *,
    cache_key: str,
    mode: str | None = None,
) -> ReviewOutcome:
    messages: list[Message] = [
        {"role": "system", "content": method.prompts["reviewer"]},
        {
            "role": "user",
            "content": review_message(c, facts, mode or method.review_mode, ctx.roots),
        },
    ]
    result = run_agent(
        model, messages, (*READ_TOOLS, SUBMIT_VERDICT), ctx, caps, cache_key=cache_key
    )
    if result.submitted is None:
        return ReviewOutcome(None, (), result)
    verdict = Verdict.from_dict(result.submitted)
    return ReviewOutcome(verdict, check_evidence(verdict, ctx.roots), result)


# ----- comparison among mergeable patches -----------------------------------------------


@dataclass(frozen=True)
class Comparison:
    pick: int | None
    reasons: str
    loop: LoopResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "pick": self.pick,
            "reasons": self.reasons,
            "stop": str(self.loop.stop),
            "turns": self.loop.turns,
            "error": self.loop.error,
        }


def compare_message(items: list[tuple[Candidate, Facts, Verdict]]) -> str:
    parts = [
        "Each patch below was reviewed on its own and judged mergeable, and none is both "
        "faster and smaller than another. Pick the one to propose. Each patched tree is "
        "readable at the root named in its heading.",
    ]
    for c, f, v in items:
        parts += [
            "",
            f"## Attempt {c.number} (root patched_{c.dirname})",
            "```diff",
            (c.patch or "").rstrip(),
            "```",
            facts_block(f),
            "",
            inputs_table(c),
            tests_line(c),
            "",
            f"Review: cost before: {v.cost_before}; after: {v.cost_after}. {v.reasons}",
        ]
    return "\n".join(parts)


def compare_candidates(
    model: ChatModel,
    method: WriterMethod,
    items: list[tuple[Candidate, Facts, Verdict]],
    ctx: ToolContext,
    caps: Caps,
    *,
    cache_key: str,
) -> Comparison:
    numbers = {c.number for c, _, _ in items}

    def validate(args: dict[str, Any]) -> str | None:
        if args.get("attempt") not in numbers:
            return f"attempt must be one of {sorted(numbers)}"
        if not isinstance(args.get("reasons"), str) or not args["reasons"].strip():
            return "reasons must be a non empty string"
        return None

    submit = submit_tool(
        "submit_pick",
        "Submit the attempt number to propose and why.",
        {
            "type": "object",
            "properties": {"attempt": {"type": "integer"}, "reasons": {"type": "string"}},
            "required": ["attempt", "reasons"],
        },
        validate,
    )
    messages: list[Message] = [
        {"role": "system", "content": method.prompts["compare"]},
        {"role": "user", "content": compare_message(items)},
    ]
    result = run_agent(model, messages, (*READ_TOOLS, submit), ctx, caps, cache_key=cache_key)
    if result.submitted is None:
        return Comparison(None, "", result)
    return Comparison(int(result.submitted["attempt"]), str(result.submitted["reasons"]), result)
