"""Pure functions over unified diff text: which files it touches, whether they are
allowed, how big it is, and a hash that ignores cosmetic differences.

Nothing here runs git. The referee applies patches inside a box; this module only
reads the text the worker produced.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)$", re.MULTILINE)
_PLUS_FILE = re.compile(r"^\+\+\+ (?:b/)?(?P<path>\S+)", re.MULTILINE)
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_INDEX = re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+")


def changed_files(diff: str) -> tuple[str, ...]:
    """Paths touched by the diff, repo relative, in order of first appearance.

    Reads ``diff --git`` headers first. Falls back to ``+++`` lines for diffs
    produced without git. A deleted file shows ``+++ /dev/null`` and is taken from
    the header instead.
    """
    seen: list[str] = []
    for m in _DIFF_HEADER.finditer(diff):
        path = m.group("b")
        if path not in seen:
            seen.append(path)
    if seen:
        return tuple(seen)
    for m in _PLUS_FILE.finditer(diff):
        path = m.group("path")
        if path != "/dev/null" and path not in seen:
            seen.append(path)
    return tuple(seen)


def _glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a path glob where ``**`` crosses directories and ``*`` does not."""
    out = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def matches_any(path: str, globs: tuple[str, ...]) -> bool:
    return any(_glob_to_regex(g).match(path) for g in globs)


@dataclass(frozen=True)
class ScopeViolation:
    path: str
    reason: str


def scope_violations(
    files: tuple[str, ...], allow: tuple[str, ...], deny: tuple[str, ...]
) -> tuple[ScopeViolation, ...]:
    """Every file that is denied, or not allowed. Empty means the patch is in scope."""
    out: list[ScopeViolation] = []
    for path in files:
        if matches_any(path, deny):
            out.append(ScopeViolation(path, "matches a denied pattern"))
        elif not matches_any(path, allow):
            out.append(ScopeViolation(path, "matches no allowed pattern"))
    return tuple(out)


def changed_line_count(diff: str) -> int:
    """Added plus removed lines, excluding file headers."""
    n = 0
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            n += 1
    return n


def added_lines(diff: str) -> frozenset[str]:
    """The distinct added lines of a diff, stripped, without the ``+++`` headers."""
    return frozenset(
        stripped
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
        if (stripped := line[1:].strip())
    )


def overlap(diff_a: str, diff_b: str) -> float | None:
    """How much two diffs share, as the fraction of the smaller one's added lines
    that the other also adds. Symmetric.

    None when either diff adds nothing, so a patch that only deletes overlaps
    nothing. Two known edges: a one line patch that happens to share that line
    with a large one scores 1.0, and the measure ignores where a line lands,
    so it reads the mechanism, not the placement.
    """
    a, b = added_lines(diff_a), added_lines(diff_b)
    if not a or not b:
        return None
    return len(a & b) / min(len(a), len(b))


def normalised_hash(diff: str) -> str:
    """A hash that is the same for two diffs that make the same change.

    Ignores ``index`` lines, hunk header line numbers, and trailing whitespace, so a
    worker that reproduces an earlier patch against a slightly shifted file is
    recognised as a duplicate.
    """
    kept: list[str] = []
    for line in diff.splitlines():
        if _INDEX.match(line):
            continue
        if _HUNK.match(line):
            kept.append("@@")
            continue
        kept.append(line.rstrip())
    return hashlib.sha256("\n".join(kept).encode()).hexdigest()[:16]
