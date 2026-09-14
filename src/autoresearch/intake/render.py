"""An accepted proposal as config text, parsed back before anyone writes it.

The target comes from the draft and the proposal; every other section from the
template config. The text is rendered fresh rather than copied from the
template, so the check is exact: the parsed result must equal the config that
was meant, field for field.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, replace
from typing import Any

from autoresearch.config import (
    BenchmarkInput,
    ConfigError,
    RunConfig,
    SuitePaths,
    TargetSpec,
    parse_config,
)
from autoresearch.intake.derive import Draft

COMMENT_WIDTH = 88


class RenderError(ValueError):
    """A proposal that does not become a config which loads as intended."""


@dataclass(frozen=True)
class ProposedInput:
    name: str
    setup: str
    regime: str
    why: str


@dataclass(frozen=True)
class Proposal:
    hot_file: str
    alias: str
    call: str
    tests_module: str
    extra_pip: tuple[str, ...]
    axis: str
    axis_reason: str
    inputs: tuple[ProposedInput, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _s(value: str) -> str:
    """A TOML basic string. JSON's escapes are a subset of TOML's."""
    return json.dumps(value, ensure_ascii=False)


def _list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_s(v) for v in values) + "]"


def normal_setup(setup: str) -> str:
    """Setup as it is written and read back: no leading blank lines, one trailing newline."""
    return setup.strip("\n") + "\n"


def _setup_value(setup: str) -> str:
    body = normal_setup(setup)
    literal_ok = "'''" not in body and all(c in "\n\t" or " " <= c != "\x7f" for c in body)
    return f"'''\n{body}'''" if literal_ok else _s(body)


def _comment(text: str, prefix: str = "") -> list[str]:
    flat = " ".join(text.split())
    width = COMMENT_WIDTH - 2 - len(prefix)
    wrapped = textwrap.wrap(flat, width) or [""]
    return [f"# {prefix}{wrapped[0]}"] + [f"# {' ' * len(prefix)}{w}" for w in wrapped[1:]]


def pip_for(draft: Draft, proposal: Proposal) -> tuple[str, ...]:
    out = list(draft.pip)
    out += [p for p in proposal.extra_pip if p not in out]
    return tuple(out)


def target_spec(draft: Draft, proposal: Proposal) -> TargetSpec:
    return TargetSpec(
        name=draft.name,
        repo=draft.repo,
        sha=draft.sha,
        package=draft.package,
        alias=proposal.alias,
        package_root=draft.package_root,
        hot_file=proposal.hot_file,
        call=proposal.call,
        tests=SuitePaths(module=proposal.tests_module, full=draft.tests_full),
        inputs=tuple(
            BenchmarkInput(name=i.name, setup=normal_setup(i.setup), noise_floor=None)
            for i in proposal.inputs
        ),
        allow=draft.allow,
        deny=draft.deny,
        pip=pip_for(draft, proposal),
        apt=(),
        docs=(),
    )


def render(draft: Draft, proposal: Proposal, template: RunConfig, when: str) -> str:
    """The config text. Raises RenderError unless it parses back to exactly what was meant."""
    t = target_spec(draft, proposal)
    lines = [
        f"# {draft.name}: written by autoresearch intake propose at {when}.",
        f"# Run `autoresearch next configs/{draft.run_id}.toml` for the step it is at.",
        "#",
        *_comment(proposal.axis, "input axis: "),
        *_comment(proposal.axis_reason, "why it matters: "),
        "",
        "[run]",
        f"run_id = {_s(draft.run_id)}",
        f"width = {template.width}",
        f"rounds = {template.rounds}",
        "",
        "[target]",
        f"name = {_s(t.name)}",
        f"repo = {_s(t.repo)}",
        f"sha = {_s(t.sha)}",
        f"package = {_s(t.package)}",
        f"alias = {_s(t.alias)}",
        f"package_root = {_s(t.package_root)}",
        f"hot_file = {_s(t.hot_file)}",
        f"call = {_s(t.call)}",
        f"pip = {_list(t.pip)}",
        f"allow = {_list(t.allow)}",
        f"deny = {_list(t.deny)}",
        "docs = []",
        "",
        "[target.tests]",
        f"module = {_s(t.tests.module)}",
        f"full = {_s(t.tests.full)}",
    ]
    for proposed in proposal.inputs:
        lines += [
            "",
            *_comment(proposed.regime, "regime: "),
            *_comment(proposed.why, "why: "),
            "[[target.inputs]]",
            f"name = {_s(proposed.name)}",
            f"setup = {_setup_value(proposed.setup)}",
        ]
    w, r, b, st, ob = (
        template.worker,
        template.referee,
        template.boxes,
        template.storage,
        template.observe,
    )
    lines += [
        "",
        "[worker]",
        f"model = {_s(w.model)}",
        f"reasoning_effort = {_s(w.reasoning_effort)}",
        f"completion_window = {_s(w.completion_window)}",
        f"max_turns = {w.max_turns}",
        f"max_seconds = {w.max_seconds}",
        f"max_input_tokens = {w.max_input_tokens}",
        f"turn_timeout = {w.turn_timeout}",
        "",
        "[referee]",
        f"pairs = {r.pairs}",
        f"repeats_per_launch = {r.repeats_per_launch}",
        f"min_clean_pairs = {r.min_clean_pairs}",
        f"hash_seeds = [{', '.join(str(s) for s in r.hash_seeds)}]",
        "",
        "[boxes]",
        f"worker_size = {_s(b.worker_size)}",
        f"referee_size = {_s(b.referee_size)}",
        f"control_size = {_s(b.control_size)}",
        f"disk_gib = {b.disk_gib}",
        "",
        "[storage]",
        f"volume = {_s(st.volume)}",
        f"mount = {_s(st.mount)}",
        "",
        "[observe]",
        f"enabled = {'true' if ob.enabled else 'false'}",
        f"session_prefix = {_s(ob.session_prefix)}",
    ]
    text = "\n".join(lines) + "\n"
    try:
        parsed = parse_config(text)
    except ConfigError as e:
        raise RenderError(f"the config does not load: {e}") from e
    meant = replace(template, run_id=draft.run_id, target=t, source_text="")
    if replace(parsed, source_text="") != meant:
        raise RenderError("the rendered config does not read back as the proposal")
    return text
