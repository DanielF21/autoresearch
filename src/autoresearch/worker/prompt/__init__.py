"""What the worker is told.

The system prompt has versions, one module each (``v1`` and on), sharing their
mechanics through ``common``. A run names the versions its worker slots use in
``[worker] prompts``; slot ``w`` gets ``prompts[w % len(prompts)]``. The rest of
the conversation, rendered by ``render``, is the same for every version.
"""

from __future__ import annotations

from autoresearch.worker.prompt import v1, v2, v3, v4
from autoresearch.worker.prompt.render import (
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

VERSIONS: dict[str, str] = {m.NAME: m.SYSTEM_PROMPT for m in (v1, v2, v3, v4)}
DEFAULT_VERSION = v1.NAME


def system_prompt(version: str, python: str) -> str:
    """The system message for one version. ``python`` is the version the box
    reported, never assumed. An unknown version is a programming error here; the
    config refuses it by name first."""
    return VERSIONS[version].format(python=python)


__all__ = [
    "DEFAULT_VERSION",
    "LAST_TURN",
    "NUDGE",
    "OUTCOME_CHARS",
    "SUMMARY_CHARS",
    "VERSIONS",
    "index_line",
    "initial_user_message",
    "mechanism_line",
    "outcome",
    "render_docs",
    "render_history",
    "render_target",
    "system_prompt",
]
