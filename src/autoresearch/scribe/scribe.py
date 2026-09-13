"""One conversation with one model: pick a candidate, then write its pull request.

The model first sees the shortlisted candidates and calls ``submit_pick`` with an
attempt number, or null when none should be merged. In the same conversation it
then reads the repository's last merged pull requests by people and calls
``submit_pr``. In both steps it may read the target's source at the base commit
with the intake's read only tools. The system prompt is ``prompt.md``, edited by
hand. A body stating a number with a unit that the measurements do not contain is
refused. Every message sent and received is saved next to the result, and a pick
also writes ``pr.md`` and the picked ``patch.diff``.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch import history
from autoresearch.config import WorkerConfig
from autoresearch.intake.propose import READ_TOOL_SPECS, READ_TOOLS, ToolError
from autoresearch.model.protocol import ChatModel, Message, ToolSpec
from autoresearch.scribe.candidates import Candidate, RunTarget, describe
from autoresearch.scribe.prs import MergedPR
from autoresearch.types import Usage

PROMPT = Path(__file__).parent / "prompt.md"
TRIES = 3
# Chosen, not measured: enough to read a hot file and a few callers in each step.
MAX_READS = 30
# A number followed by a unit. Bare numbers are not checked: sizes, line numbers and
# constants in the body come from the source, not the measurements.
UNIT_NUMBER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s?(?:x|\u00d7|%|ms|\u00b5s|us|seconds|second|s|times)(?!\w)"
)
ANY_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# No hyphen or dash anywhere in the prose of a title or body; code may keep them.
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-*:?\s*$")
LIST_MARKER = re.compile(r"^\s*[-*+]\s+")
DASHED = re.compile(r"\S*[-\u2010-\u2015]\S*")

Validator = Callable[[dict[str, Any]], str | None]


class ScribeError(RuntimeError):
    """The model gave no valid answer within ``TRIES`` replies."""


def worker_config(run_dir: Path, model: str | None = None) -> WorkerConfig:
    """The run's own worker model settings, with an optional model override."""
    data = tomllib.loads((run_dir / history.CONFIG_FILE).read_text())
    w = data.get("worker")
    if not isinstance(w, dict):
        raise ScribeError(f"{run_dir / history.CONFIG_FILE} has no [worker] table")
    return WorkerConfig(
        model=model or str(w["model"]),
        reasoning_effort=str(w["reasoning_effort"]),
        completion_window=str(w["completion_window"]),
        max_turns=int(w.get("max_turns", 0)),
        max_seconds=int(w.get("max_seconds", 0)),
        max_input_tokens=int(w.get("max_input_tokens", 0)),
        turn_timeout=int(w["turn_timeout"]),
    )


def _tool(name: str, description: str, properties: dict[str, Any]) -> ToolSpec:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }


SUBMIT_PICK = _tool(
    "submit_pick",
    "The attempt number to propose, or null when no maintainer should merge any of them, and why.",
    {"attempt": {"type": ["integer", "null"]}, "reasons": {"type": "string"}},
)
SUBMIT_PR = _tool(
    "submit_pr",
    "The pull request title and its Markdown body.",
    {"title": {"type": "string"}, "body": {"type": "string"}},
)


@dataclass
class Conversation:
    model: ChatModel
    cache_key: str
    source: Path | None = None
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def ask(self, content: str, tool: ToolSpec, validate: Validator) -> dict[str, Any]:
        """Send ``content`` and return the first valid call of ``tool``.

        Read calls against ``source`` are answered and do not count as misses, up to
        ``MAX_READS`` in this step. A reply with no tool call, a call that fails
        validation, or a reply whose reads were all refused is a miss, and the problem
        goes back as text. After ``TRIES`` misses, raises ScribeError.
        """
        name = tool["function"]["name"]
        tools = [tool] if self.source is None else [tool, *READ_TOOL_SPECS]
        names = [t["function"]["name"] for t in tools]
        self.messages.append({"role": "user", "content": content})
        misses = reads = 0
        while misses < TRIES:
            resp = self.model.complete(self.messages, tools, cache_key=self.cache_key)
            self.usage = self.usage + resp.usage
            self.messages.append(resp.message)
            if not resp.tool_calls:
                misses += 1
                self.messages.append({"role": "user", "content": f"Answer by calling {name}."})
                continue
            accepted: dict[str, Any] | None = None
            read_something = False
            for call in resp.tool_calls:
                if call.name == name:
                    problem = validate(call.arguments)
                    if problem is None and accepted is None:
                        accepted = call.arguments
                    reply = "accepted" if problem is None else f"not accepted: {problem}"
                elif self.source is not None and call.name in READ_TOOLS:
                    if reads >= MAX_READS:
                        reply = f"not run: all {MAX_READS} reads are used; call {name} now"
                    else:
                        reads += 1
                        read_something = True
                        try:
                            reply = READ_TOOLS[call.name](self.source, call.arguments)
                        except (ToolError, ValueError, OSError) as e:
                            reply = f"error: {e}"
                else:
                    reply = f"not accepted: the tools here are {', '.join(names)}"
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
            if accepted is not None:
                return accepted
            if not read_something:
                misses += 1
        raise ScribeError(f"no valid {name} call in {TRIES} replies")


def pick_message(target: RunTarget, kept: list[Candidate]) -> str:
    parts = [
        f"Repository: {target.repo}",
        f"Base commit every patch was written against and measured on: {target.sha}",
        "",
        f"{len(kept)} candidate patch(es) passed the filter. Each one applied, passed the "
        "tests, computed the same result as the base on every input, cleared the noise "
        "floor and was slower on no input. Each input was timed in alternating pairs of "
        "base and patched runs on one machine.",
        "",
        "Pick the one to propose as a pull request, or none if no maintainer should merge "
        "any of them. Call submit_pick.",
    ]
    for c in kept:
        parts += ["", describe(c)]
    return "\n".join(parts)


def write_message(attempt: int, prs: list[MergedPR]) -> str:
    parts = [
        f"Write the pull request for attempt {attempt}.",
        f"Below are the last {len(prs)} pull requests merged in this repository by people. "
        "Write the title and description the way these are written. Call submit_pr.",
    ]
    for pr in prs:
        parts += ["", "---", "", f"# {pr.title}", "", pr.body.strip() or "(empty body)"]
    return "\n".join(parts)


def _validate_pick(numbers: set[int]) -> Validator:
    def validate(args: dict[str, Any]) -> str | None:
        attempt = args.get("attempt")
        valid = isinstance(attempt, int) and not isinstance(attempt, bool) and attempt in numbers
        if attempt is not None and not valid:
            return f"attempt must be one of {sorted(numbers)} or null"
        reasons = args.get("reasons")
        if not isinstance(reasons, str) or not reasons.strip():
            return "reasons must be a non empty string"
        return None

    return validate


def unsupported_numbers(text_: str, evidence: str) -> list[str]:
    """Numbers with a unit in ``text_`` that no number in ``evidence`` equals, as written or rounded.

    ``6.2x`` is supported by a measured 6.23; ``7.1x`` is not, unless some number in
    the evidence rounds to 7.1.
    """
    known = {float(m.group(0)) for m in ANY_NUMBER.finditer(evidence)}
    missing: list[str] = []
    for m in UNIT_NUMBER.finditer(text_):
        written = m.group(1)
        places = len(written.split(".")[1]) if "." in written else 0
        value = f"{float(written):.{places}f}"
        if not any(f"{k:.{places}f}" == value for k in known):
            token = m.group(0).strip()
            if token not in missing:
                missing.append(token)
    return missing


def prose_dashes(text_: str) -> list[str]:
    """Every word of prose in ``text_`` that holds a hyphen or a dash, in order, once each.

    Code blocks, code in backticks, a Markdown table's separator row and a list
    item's leading marker are not prose. Everything else is, so ``full-graph``, ``->``
    and a lone em dash are all found.
    """
    prose = INLINE_CODE.sub(" ", CODE_BLOCK.sub(" ", text_))
    found: list[str] = []
    for line in prose.splitlines():
        if TABLE_RULE.match(line):
            continue
        for word in DASHED.findall(LIST_MARKER.sub("", line, count=1)):
            if word not in found:
                found.append(word)
    return found


def _validate_pr(evidence: str) -> Validator:
    def validate(args: dict[str, Any]) -> str | None:
        for key in ("title", "body"):
            value = args.get(key)
            if not isinstance(value, str) or not value.strip():
                return f"{key} must be a non empty string"
        text_ = f"{args['title']}\n{args['body']}"
        problems: list[str] = []
        missing = unsupported_numbers(text_, evidence)
        if missing:
            problems.append(
                f"these numbers are not in the measurements you were shown: {', '.join(missing)}. "
                "State measured numbers only, as shown or rounded."
            )
        dashed = prose_dashes(text_)
        if dashed:
            problems.append(
                f"these words hold a hyphen or a dash: {', '.join(dashed)}. Use none in the "
                "title or the prose, even where grammar calls for one: join the words with a "
                "space instead. Only code in backticks may contain one."
            )
        return " ".join(problems) or None

    return validate


@dataclass(frozen=True)
class Draft:
    pick: int | None
    reasons: str
    title: str
    body: str


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(
    model: ChatModel,
    target: RunTarget,
    kept: list[Candidate],
    prs: list[MergedPR],
    out: Path,
    *,
    prompt: str | None = None,
    source: Path | None = None,
) -> Draft:
    """The pick, then the pull request, written under ``out`` as they arrive."""
    convo = Conversation(model, cache_key=f"scribe-{target.run_id}", source=source)
    system = PROMPT.read_text() if prompt is None else prompt
    convo.messages.append({"role": "system", "content": system})
    evidence = pick_message(target, kept)
    try:
        picked = convo.ask(evidence, SUBMIT_PICK, _validate_pick({c.number for c in kept}))
        attempt = None if picked["attempt"] is None else int(picked["attempt"])
        reasons = str(picked["reasons"]).strip()
        _write_json(out / "pick.json", {"attempt": attempt, "reasons": reasons})
        if attempt is None:
            return Draft(None, reasons, "", "")
        pr = convo.ask(write_message(attempt, prs), SUBMIT_PR, _validate_pr(evidence))
        title, body = str(pr["title"]).strip(), str(pr["body"]).strip()
        (out / "title.txt").write_text(title + "\n")
        (out / "body.md").write_text(body + "\n")
        (out / "pr.md").write_text(f"# {title}\n\n{body}\n")
        patch = next(c.patch for c in kept if c.number == attempt)
        (out / "patch.diff").write_text(patch or "")
        return Draft(attempt, reasons, title, body)
    finally:
        _write_json(out / "messages.json", convo.messages)
        _write_json(out / "usage.json", convo.usage.to_dict())
