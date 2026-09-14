"""What the worker is told.

The system prompt has versions, one module each, sharing their mechanics through
``common``: v1 and on are composed from it, v0 is the width experiment's prompt
kept verbatim from before it. A run names the versions its worker slots use in
``[worker] prompts``; slot ``w`` gets ``prompts[w % len(prompts)]``. The rest of
the conversation, rendered by ``render``, is the same for every version except
the closing sentence of the first message, which v0 keeps from its time.
"""

from __future__ import annotations

from autoresearch.worker.prompt import common, v0, v1, v2, v3, v4
from autoresearch.worker.prompt.render import (
    CLOSING,
    LAST_TURN,
    NUDGE,
    OUTCOME_CHARS,
    SUMMARY_CHARS,
    index_line,
    initial_user_message,
    mechanism_line,
    outcome,
    render_docs,
    render_history,
    render_target,
)

VERSIONS: dict[str, str] = {m.NAME: m.SYSTEM_PROMPT for m in (v0, v1, v2, v3, v4)}
DEFAULT_VERSION = v1.NAME

# Closing sentences that differ from ``render.CLOSING``. Only v0 has one.
CLOSINGS: dict[str, str] = {v0.NAME: v0.CLOSING}


def closing(version: str) -> str:
    """The first message's closing sentence for one version."""
    return CLOSINGS.get(version, CLOSING)


def system_prompt(version: str, python: str, history_index: bool = True) -> str:
    """The system message for one version. ``python`` is the version the box
    reported, never assumed. ``history_index`` picks the sentence the sandbox
    block ends on; v0 has no placeholder for it and ignores the argument. An
    unknown version is a programming error here; the config refuses it by name
    first."""
    note = common.INDEX_NOTE if history_index else common.NO_INDEX_NOTE
    return VERSIONS[version].format(python=python, index_note=note)


__all__ = [
    "CLOSING",
    "CLOSINGS",
    "DEFAULT_VERSION",
    "LAST_TURN",
    "NUDGE",
    "OUTCOME_CHARS",
    "SUMMARY_CHARS",
    "VERSIONS",
    "closing",
    "index_line",
    "initial_user_message",
    "mechanism_line",
    "outcome",
    "render_docs",
    "render_history",
    "render_target",
    "system_prompt",
]
