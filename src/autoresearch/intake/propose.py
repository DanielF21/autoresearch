"""One model conversation that chooses the call and the inputs, with read only tools.

The model reads the clone through ``list_files``, ``read_file`` and ``grep``,
all plain Python over the file tree, confined to the clone and executing
nothing. It answers with ``submit_proposal``. A proposal is validated without
running anything, rendered into config text and parsed back; any problem goes
back to the model as the tool's reply. Every message and the token usage are
saved beside the draft whatever happens.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch.config import SEED_NAME, BenchmarkInput, RunConfig
from autoresearch.intake.derive import REPO_DIR, Draft
from autoresearch.intake.render import Proposal, ProposedInput, RenderError, render
from autoresearch.model.protocol import ChatModel, Message, ToolSpec
from autoresearch.types import Usage

PROMPT = Path(__file__).parent / "prompt.md"
MAX_OUTPUT = 20_000
READ_MAX_LINES = 400
LIST_MAX = 400
GREP_MAX = 200
GREP_FILE_BYTES = 2_000_000
# Chosen limits, not measured ones. Two because one input is what generality.md
# showed fails; twelve because every input costs six timing pairs per attempt.
MIN_INPUTS = 2
MAX_INPUTS = 12
WARN_TURNS_LEFT = 3
INPUT_NAME = re.compile(r"^[A-Za-z0-9_]+$")
PIP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProposeError(RuntimeError):
    """No proposal was accepted within the turn limit."""


class ToolError(ValueError):
    """A tool call the model can correct: the message goes back as the reply."""


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    half = MAX_OUTPUT // 2
    return f"{text[:half]}\n... [{len(text) - MAX_OUTPUT} characters cut] ...\n{text[-half:]}"


def _inside(repo: Path, path: str) -> Path:
    root = repo.resolve()
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ToolError(f"{path} is outside the repository")
    if ".git" in p.relative_to(root).parts:
        raise ToolError("the .git directory is not readable here")
    return p


def _rel(repo: Path, p: Path) -> str:
    return str(p.relative_to(repo.resolve()))


def _walk(repo: Path, start: Path, glob: str) -> list[Path]:
    """Files under ``start`` matching ``glob``, never leaving the repository."""
    root = repo.resolve()
    if start.is_file():
        return [start]
    out: list[Path] = []
    for p in sorted(start.glob(glob)):
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved.is_file() and root in resolved.parents and ".git" not in p.parts:
            out.append(p)
    return out


def list_files(repo: Path, args: dict[str, Any]) -> str:
    path = str(args.get("path") or ".")
    glob = str(args.get("glob") or "*")
    start = _inside(repo, path)
    if not start.is_dir():
        raise ToolError(f"{path} is not a directory")
    rows: list[str] = []
    for p in sorted(start.glob(glob)):
        if ".git" in p.relative_to(start).parts or p.name == ".git":
            continue
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved != repo.resolve() and repo.resolve() not in resolved.parents:
            continue
        if p.is_dir():
            rows.append(f"{_rel(repo, resolved)}/")
        else:
            rows.append(f"{_rel(repo, resolved)}  {p.stat().st_size} bytes")
        if len(rows) >= LIST_MAX:
            rows.append(f"... stopped at {LIST_MAX} entries; narrow path or glob")
            break
    return "\n".join(rows) or "no matches"


def read_file(repo: Path, args: dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    p = _inside(repo, path)
    if not p.is_file():
        raise ToolError(f"{path} is not a file")
    start = max(1, int(args.get("start") or 1))
    count = min(READ_MAX_LINES, max(1, int(args.get("count") or READ_MAX_LINES)))
    lines = p.read_text(errors="replace").splitlines()
    chunk = lines[start - 1 : start - 1 + count]
    body = "\n".join(f"{n:>6}  {line}" for n, line in enumerate(chunk, start))
    tail = f"\n[lines {start} to {start + len(chunk) - 1} of {len(lines)}]"
    return _clip(body + tail)


def grep(repo: Path, args: dict[str, Any]) -> str:
    try:
        pattern = re.compile(str(args.get("pattern") or ""))
    except re.error as e:
        raise ToolError(f"pattern does not compile: {e}") from e
    start = _inside(repo, str(args.get("path") or "."))
    glob = str(args.get("glob") or "**/*.py")
    hits: list[str] = []
    for p in _walk(repo, start, glob):
        if p.stat().st_size > GREP_FILE_BYTES:
            continue
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{_rel(repo, p.resolve())}:{n}: {line.strip()[:300]}")
                if len(hits) >= GREP_MAX:
                    return _clip("\n".join(hits) + f"\n... stopped at {GREP_MAX} matches")
    return _clip("\n".join(hits)) or "no matches"


def _tool(name: str, description: str, props: dict[str, Any], required: list[str]) -> ToolSpec:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


READ_TOOLS: dict[str, Callable[[Path, dict[str, Any]], str]] = {
    "list_files": list_files,
    "read_file": read_file,
    "grep": grep,
}

READ_TOOL_SPECS: list[ToolSpec] = [
    _tool(
        "list_files",
        f"List files under a directory of the repository, with sizes. glob defaults to '*'; "
        f"'**/*.py' recurses. At most {LIST_MAX} entries.",
        {"path": {"type": "string"}, "glob": {"type": "string"}},
        [],
    ),
    _tool(
        "read_file",
        f"Read lines of a file, numbered. start is 1 based; count is at most {READ_MAX_LINES}.",
        {"path": {"type": "string"}, "start": {"type": "integer"}, "count": {"type": "integer"}},
        ["path"],
    ),
    _tool(
        "grep",
        f"Search files for a Python regular expression. path is a file or directory; glob "
        f"defaults to '**/*.py'. At most {GREP_MAX} matches.",
        {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}},
        ["pattern"],
    ),
]

TOOL_SPECS: list[ToolSpec] = [
    *READ_TOOL_SPECS,
    _tool(
        "submit_proposal",
        "The benchmark: hot file, alias, call, the hot file's own tests, "
        "any extra pip packages setup needs, the input axis and why it matters, and the inputs.",
        {
            "hot_file": {"type": "string"},
            "alias": {"type": "string"},
            "call": {"type": "string"},
            "tests_module": {"type": "string"},
            "extra_pip": {"type": "array", "items": {"type": "string"}},
            "axis": {"type": "string"},
            "axis_reason": {"type": "string"},
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "setup": {"type": "string"},
                        "regime": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["name", "setup", "regime", "why"],
                },
            },
        },
        [
            "hot_file",
            "alias",
            "call",
            "tests_module",
            "extra_pip",
            "axis",
            "axis_reason",
            "inputs",
        ],
    ),
]


def _parses(source: str, mode: str) -> str:
    try:
        ast.parse(source, mode=mode)
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    return ""


def _not_one_call(call: str) -> str:
    """Why ``call`` is not a single function call, or empty.

    The result the referee fingerprints has to be what the library computed.
    The first pycodestyle proposal under the seed rule wrote
    ``[style.input_file('stdin', lines=source), source_tag]``: a list carrying
    a per seed tag beside a count that was zero at every seed, so two seeds
    fingerprinted differently while the library's answer never changed. One
    call, with no display or operator around it, leaves nothing to smuggle.
    """
    node = ast.parse(call, mode="eval").body
    if isinstance(node, ast.Call):
        return ""
    return (
        "call must be one function call, such as pkg.f(x) or list(pkg.g(x)); a list, tuple, "
        "dict or expression around it can carry setup values into the result, and the "
        "result must be what the library computed"
    )


def _text(args: dict[str, Any], key: str, problems: list[str], *, empty: bool = False) -> str:
    value = args.get(key)
    if not isinstance(value, str) or (not empty and not value.strip()):
        problems.append(f"{key} must be a {'' if empty else 'non empty '}string")
        return ""
    return value.strip()


def validate(
    args: dict[str, Any], repo: Path, draft: Draft, template: RunConfig, when: str
) -> tuple[Proposal, str] | str:
    """The proposal and its config text, or every problem found, as one message."""
    problems: list[str] = []
    hot_file = _text(args, "hot_file", problems)
    alias = _text(args, "alias", problems)
    call = _text(args, "call", problems)
    tests_module = _text(args, "tests_module", problems)
    axis = _text(args, "axis", problems)
    axis_reason = _text(args, "axis_reason", problems)

    if hot_file:
        try:
            p = _inside(repo, hot_file)
        except ToolError as e:
            problems.append(f"hot_file: {e}")
        else:
            if not (p.is_file() and p.suffix == ".py"):
                problems.append(f"hot_file {hot_file} is not a .py file in the repository")
            elif draft.single_module:
                if hot_file != draft.package_dir:
                    problems.append(f"hot_file must be the single module, {draft.package_dir}")
            elif not hot_file.startswith(f"{draft.package_dir}/"):
                problems.append(f"hot_file must be inside the package, {draft.package_dir}/")
    if tests_module:
        try:
            if not _inside(repo, tests_module).exists():
                problems.append(f"tests_module {tests_module} does not exist")
        except ToolError as e:
            problems.append(f"tests_module: {e}")
    if alias and (not alias.isidentifier() or alias == "ROOT"):
        problems.append("alias must be a Python identifier other than ROOT")
    if call and (err := _parses(call, "eval")):
        problems.append(f"call is not one expression: {err}")
    elif call and (err := _not_one_call(call)):
        problems.append(err)

    extra = args.get("extra_pip")
    if not isinstance(extra, list) or not all(
        isinstance(x, str) and PIP_NAME.match(x) for x in extra
    ):
        problems.append("extra_pip must be a list of package names")
        extra = []

    raw_inputs = args.get("inputs")
    inputs: list[ProposedInput] = []
    if not isinstance(raw_inputs, list) or not MIN_INPUTS <= len(raw_inputs) <= MAX_INPUTS:
        problems.append(f"inputs must be a list of {MIN_INPUTS} to {MAX_INPUTS} inputs")
        raw_inputs = []
    seen: set[str] = set()
    for n, raw in enumerate(raw_inputs):
        if not isinstance(raw, dict):
            problems.append(f"inputs[{n}] must be an object")
            continue
        fields = {k: raw.get(k) for k in ("name", "setup", "regime", "why")}
        if not all(isinstance(v, str) and v.strip() for v in fields.values()):
            problems.append(f"inputs[{n}] needs non empty name, setup, regime and why")
            continue
        name = str(fields["name"]).strip()
        if not INPUT_NAME.match(name):
            problems.append(f"input name {name!r} may use only letters, digits and underscores")
        if name in seen:
            problems.append(f"input name {name!r} appears twice")
        seen.add(name)
        if err := _parses(str(fields["setup"]), "exec"):
            problems.append(f"setup of {name} does not parse: {err}")
        elif not BenchmarkInput(name, str(fields["setup"]), None).seeded:
            problems.append(
                f"setup of {name} does not read {SEED_NAME}; the referee times an instance "
                "the worker was not shown, so every setup must build its input from it"
            )
        inputs.append(
            ProposedInput(
                name=name,
                setup=str(fields["setup"]),
                regime=str(fields["regime"]).strip(),
                why=str(fields["why"]).strip(),
            )
        )
    if problems:
        return "; ".join(problems)

    proposal = Proposal(
        hot_file=hot_file,
        alias=alias,
        call=call,
        tests_module=tests_module,
        extra_pip=tuple(str(x) for x in extra),
        axis=axis,
        axis_reason=axis_reason,
        inputs=tuple(inputs),
    )
    try:
        return proposal, render(draft, proposal, template, when)
    except RenderError as e:
        return str(e)


REPAIR = """\
A referee box ran the config your proposal became. Its report follows. Each FAIL line \
is a rule the config broke, and the harness will not measure the target until every one \
passes; WARN lines are advice. Read whatever code you need, then call submit_proposal \
again. You may change anything in the proposal, including the hot file, the call and \
the inputs.

{report}
"""


@dataclass
class Outcome:
    proposal: Proposal
    text: str
    turns: int
    usage: Usage = field(default_factory=Usage)


@dataclass
class _Session:
    messages: list[Message]
    usage: Usage = field(default_factory=Usage)

    def save(self, out: Path) -> None:
        (out / "messages.json").write_text(json.dumps(self.messages, indent=2) + "\n")
        (out / "usage.json").write_text(json.dumps(self.usage.to_dict(), indent=2) + "\n")


def _converse(
    model: ChatModel,
    session: _Session,
    out: Path,
    draft: Draft,
    template: RunConfig,
    *,
    when: str,
    max_turns: int,
) -> Outcome:
    """Converse until a proposal is accepted or ``max_turns`` replies have come back."""
    repo = out / REPO_DIR
    messages = session.messages
    for turn in range(1, max_turns + 1):
        resp = model.complete(messages, TOOL_SPECS, cache_key=f"intake-{draft.name}")
        session.usage = session.usage + resp.usage
        messages.append(resp.message)
        accepted: tuple[Proposal, str] | None = None
        if not resp.tool_calls:
            messages.append(
                {"role": "user", "content": "Use the tools, then call submit_proposal."}
            )
        for call in resp.tool_calls:
            if call.malformed:
                reply = call.malformed_reply
            elif call.name == "submit_proposal":
                result = validate(call.arguments, repo, draft, template, when)
                if isinstance(result, str):
                    reply = f"not accepted: {result}"
                else:
                    reply = "accepted"
                    accepted = accepted or result
            elif call.name in READ_TOOLS:
                try:
                    reply = READ_TOOLS[call.name](repo, call.arguments)
                except (ToolError, ValueError, OSError) as e:
                    reply = f"error: {e}"
            else:
                reply = f"error: unknown tool {call.name!r}; available: {', '.join([*READ_TOOLS, 'submit_proposal'])}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
        if accepted is not None:
            return Outcome(accepted[0], accepted[1], turn, session.usage)
        left = max_turns - turn
        if left == WARN_TURNS_LEFT:
            messages.append(
                {
                    "role": "user",
                    "content": f"{left} replies left. Call submit_proposal with your best answer.",
                }
            )
    raise ProposeError(f"no proposal accepted in {max_turns} turns")


def propose(
    model: ChatModel,
    out: Path,
    draft: Draft,
    template: RunConfig,
    brief: str,
    *,
    when: str,
    max_turns: int,
    prompt: str | None = None,
) -> Outcome:
    """A new conversation: the brief, the tools, until a proposal is accepted."""
    system = PROMPT.read_text() if prompt is None else prompt
    session = _Session([{"role": "system", "content": system}, {"role": "user", "content": brief}])
    try:
        return _converse(model, session, out, draft, template, when=when, max_turns=max_turns)
    finally:
        session.save(out)


def repair(
    model: ChatModel,
    out: Path,
    draft: Draft,
    template: RunConfig,
    report: str,
    *,
    when: str,
    max_turns: int,
) -> Outcome:
    """The saved conversation, continued with a failed check's report, until a new proposal.

    Same messages and cache key, so everything the model read before is cached input.
    The saved usage is carried forward, so ``usage.json`` stays the conversation's total.
    """
    messages: list[Message] = json.loads((out / "messages.json").read_text())
    usage_path = out / "usage.json"
    usage = Usage.from_dict(json.loads(usage_path.read_text())) if usage_path.is_file() else Usage()
    session = _Session(
        [*messages, {"role": "user", "content": REPAIR.format(report=report)}], usage
    )
    try:
        return _converse(model, session, out, draft, template, when=when, max_turns=max_turns)
    finally:
        session.save(out)
