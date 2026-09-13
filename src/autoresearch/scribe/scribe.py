"""One conversation with one model: pick a candidate, then write its pull request.

The model first sees every candidate that passed the filter and calls
``submit_pick`` with an attempt number, or null when none should be merged. In the
same conversation it then reads the repository's last merged pull requests by
people and calls ``submit_pr``. The system prompt is ``prompt.md``, edited by hand.
Every message sent and received is saved next to the result.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch import history
from autoresearch.config import WorkerConfig
from autoresearch.model.protocol import ChatModel, Message, ToolSpec
from autoresearch.scribe.candidates import Candidate, RunTarget, describe
from autoresearch.scribe.prs import MergedPR
from autoresearch.types import Usage

PROMPT = Path(__file__).parent / "prompt.md"
TRIES = 3

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
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def ask(self, content: str, tool: ToolSpec, validate: Validator) -> dict[str, Any]:
        """Send ``content`` and return the first valid call of ``tool``.

        A reply with no tool call, or a call that fails validation, gets the problem
        back as text. After ``TRIES`` replies with no valid call, raises ScribeError.
        """
        name = tool["function"]["name"]
        self.messages.append({"role": "user", "content": content})
        for _ in range(TRIES):
            resp = self.model.complete(self.messages, [tool], cache_key=self.cache_key)
            self.usage = self.usage + resp.usage
            self.messages.append(resp.message)
            if not resp.tool_calls:
                self.messages.append({"role": "user", "content": f"Answer by calling {name}."})
                continue
            accepted: dict[str, Any] | None = None
            for call in resp.tool_calls:
                problem = (
                    validate(call.arguments)
                    if call.name == name
                    else f"the only tool here is {name}"
                )
                if problem is None and accepted is None:
                    accepted = call.arguments
                reply = "accepted" if problem is None else f"not accepted: {problem}"
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
            if accepted is not None:
                return accepted
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


def _validate_pr(args: dict[str, Any]) -> str | None:
    for key in ("title", "body"):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            return f"{key} must be a non empty string"
    return None


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
) -> Draft:
    """The pick, then the pull request, written under ``out`` as they arrive."""
    convo = Conversation(model, cache_key=f"scribe-{target.run_id}")
    system = PROMPT.read_text() if prompt is None else prompt
    convo.messages.append({"role": "system", "content": system})
    try:
        picked = convo.ask(
            pick_message(target, kept), SUBMIT_PICK, _validate_pick({c.number for c in kept})
        )
        attempt = None if picked["attempt"] is None else int(picked["attempt"])
        reasons = str(picked["reasons"]).strip()
        _write_json(out / "pick.json", {"attempt": attempt, "reasons": reasons})
        if attempt is None:
            return Draft(None, reasons, "", "")
        pr = convo.ask(write_message(attempt, prs), SUBMIT_PR, _validate_pr)
        title, body = str(pr["title"]).strip(), str(pr["body"]).strip()
        (out / "title.txt").write_text(title + "\n")
        (out / "body.md").write_text(body + "\n")
        return Draft(attempt, reasons, title, body)
    finally:
        _write_json(out / "messages.json", convo.messages)
        _write_json(out / "usage.json", convo.usage.to_dict())
