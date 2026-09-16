"""A program's files as the patch the referee measures.

Each changed file is a ``diff --git`` header followed by difflib's unified diff.
``git apply`` takes it with or without the header; ``autoresearch.patch`` and
``analysis/width/load.py`` read files from the header, so it is always written.
Both texts must end in a newline, or difflib writes a last line git refuses.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping


def unified(path: str, old: str, new: str) -> str:
    for label, text in (("base", old), ("new", new)):
        if text and not text.endswith("\n"):
            raise ValueError(f"the {label} text of {path} does not end in a newline")
    body = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return f"diff --git a/{path} b/{path}\n" + "".join(body)


def build_patch(base: Mapping[str, str], files: Mapping[str, str]) -> str | None:
    """The patch from ``base`` to ``files``, or None when nothing changed."""
    parts = [unified(p, base[p], files[p]) for p in sorted(files) if files[p] != base[p]]
    return "".join(parts) or None
