"""What may change, and applying the model's edits to it.

The paper marks evolvable code with ``# EVOLVE-BLOCK-START`` and
``# EVOLVE-BLOCK-END`` and reports evolving up to hundreds of lines. The targets'
hot files run to thousands, so the blocks are chosen by a rule, never by hand:

1. Read the rows of the flat profile document, the one the harness worker is
   shown, in descending self time.
2. Keep rows in files the target allows and does not deny.
3. Find each row's function in that file's syntax tree by the line cProfile
   reports, decorators included, and take its source lines as a block. A span
   overlapping a block already taken is skipped.
4. Stop before the block that would take the total past ``max_code_length``,
   OpenEvolve's cap on a program's characters. The first block is always taken.

A program is one text per block, in block order. Its files are the base files
with each block's lines replaced by its text, so no edited file is ever parsed
for boundaries. An edit is the paper's SEARCH/REPLACE form, matched line by line
as OpenEvolve matches it, but strictly: a SEARCH must match exactly one place
across all the blocks, or the reply yields no program.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from autoresearch.patch import matches_any

MARK_START = "# EVOLVE-BLOCK-START"
MARK_END = "# EVOLVE-BLOCK-END"

# ncalls tottime percall cumtime percall filename:lineno(function)
_ROW = re.compile(
    r"^\s*\d+(?:/\d+)?\s+(?P<tottime>\d+(?:\.\d+)?)\s+\S+\s+\S+\s+\S+\s+"
    r"(?P<path>[^\s:{}]+):(?P<line>\d+)\((?P<function>[^)]*)\)\s*$"
)
# OpenEvolve's pattern, openevolve/utils/code_utils.py ``extract_diffs``.
_EDIT = re.compile(r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE", re.DOTALL)


class BlocksError(ValueError):
    """No evolvable code could be derived."""


class EditError(ValueError):
    """A reply whose edits cannot be applied. Recorded as the attempt's error."""


@dataclass(frozen=True)
class ProfileRow:
    tottime: float
    path: str
    line: int
    function: str


@dataclass(frozen=True)
class Block:
    """Lines ``start`` to ``end`` of ``path`` at the base commit, one based and
    inclusive, and their text, which ends in a newline."""

    path: str
    start: int
    end: int
    function: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start": self.start,
            "end": self.end,
            "function": self.function,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Block:
        return cls(
            path=str(d["path"]),
            start=int(d["start"]),
            end=int(d["end"]),
            function=str(d["function"]),
            text=str(d["text"]),
        )


def profile_rows(flat: str) -> tuple[ProfileRow, ...]:
    """Every function row of a cProfile flat view, by self time, highest first."""
    rows = [
        ProfileRow(float(m["tottime"]), m["path"], int(m["line"]), m["function"])
        for line in flat.splitlines()
        if (m := _ROW.match(line))
    ]
    return tuple(sorted(rows, key=lambda r: -r.tottime))


def _in_scope(path: str, allow: tuple[str, ...], deny: tuple[str, ...]) -> bool:
    return matches_any(path, allow) and not matches_any(path, deny)


def candidate_paths(flat: str, allow: tuple[str, ...], deny: tuple[str, ...]) -> tuple[str, ...]:
    """The files the rule may draw blocks from, in the order the profile ranks them."""
    out: list[str] = []
    for row in profile_rows(flat):
        if _in_scope(row.path, allow, deny) and row.path not in out:
            out.append(row.path)
    return tuple(out)


def function_span(tree: ast.Module, line: int) -> tuple[int, int] | None:
    """The lines of the function cProfile names by ``line``. cProfile reports the
    code object's first line, which is the first decorator's when there is one."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            if line in (node.lineno, first) and node.end_lineno is not None:
                return first, node.end_lineno
    return None


def derive(
    flat: str,
    sources: Mapping[str, str],
    allow: tuple[str, ...],
    deny: tuple[str, ...],
    max_code_length: int,
) -> tuple[Block, ...]:
    """The blocks, in file and line order. See the module docstring for the rule."""
    chosen: list[Block] = []
    total = 0
    trees: dict[str, ast.Module] = {}
    for row in profile_rows(flat):
        if not _in_scope(row.path, allow, deny):
            continue
        source = sources.get(row.path)
        if source is None:
            raise BlocksError(f"no base text for {row.path}, which the profile names")
        if row.path not in trees:
            trees[row.path] = ast.parse(source)
        span = function_span(trees[row.path], row.line)
        if span is None:
            continue
        start, end = span
        if any(b.path == row.path and start <= b.end and b.start <= end for b in chosen):
            continue
        text = "".join(source.splitlines(keepends=True)[start - 1 : end])
        if chosen and total + len(text) > max_code_length:
            break
        chosen.append(Block(row.path, start, end, row.function, text))
        total += len(text)
    if not chosen:
        raise BlocksError("the profile names no function inside the allowed files")
    return tuple(sorted(chosen, key=lambda b: (b.path, b.start)))


def parse_edits(reply: str) -> tuple[tuple[str, str], ...]:
    """(search, replace) pairs, each stripped at the end as OpenEvolve strips them."""
    return tuple((s.rstrip(), r.rstrip()) for s, r in _EDIT.findall(reply))


def strip_edits(reply: str) -> str:
    """The reply's prose without its edit blocks: the attempt's rationale."""
    return _EDIT.sub("[edit]", reply).strip()


def apply_edits(texts: Sequence[str], edits: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """Apply every edit in order, each to the one place its SEARCH matches."""
    if not edits:
        raise EditError("the reply holds no SEARCH/REPLACE block")
    current = [t.split("\n") for t in texts]
    for n, (search, replace) in enumerate(edits, 1):
        if not search.strip():
            raise EditError(f"edit {n}: SEARCH is empty")
        needle = search.split("\n")
        hits = [
            (i, j)
            for i, lines in enumerate(current)
            for j in range(len(lines) - len(needle) + 1)
            if lines[j : j + len(needle)] == needle
        ]
        if not hits:
            raise EditError(f"edit {n}: SEARCH matches no marked block: {needle[0][:80]!r}")
        if len(hits) > 1:
            raise EditError(f"edit {n}: SEARCH matches {len(hits)} places; it must match one")
        i, j = hits[0]
        current[i][j : j + len(needle)] = replace.split("\n") if replace else []
    out: list[str] = []
    for lines in current:
        text = "\n".join(lines)
        out.append(text if text.endswith("\n") else text + "\n")
    return tuple(out)


def assemble(
    sources: Mapping[str, str], blocks: Sequence[Block], texts: Sequence[str]
) -> dict[str, str]:
    """Every file holding a block, with each block's lines replaced by its text."""
    pairs = list(zip(blocks, texts, strict=True))
    out: dict[str, str] = {}
    for path in sorted({b.path for b in blocks}):
        lines = sources[path].splitlines(keepends=True)
        for block, text in sorted(
            (p for p in pairs if p[0].path == path), key=lambda p: -p[0].start
        ):
            lines[block.start - 1 : block.end] = [text]
        out[path] = "".join(lines)
    return out


def marked(source: str, blocks: Sequence[Block]) -> str:
    """One file with its blocks between the paper's markers, at each block's indent."""
    lines = source.splitlines(keepends=True)
    for block in sorted(blocks, key=lambda b: -b.start):
        first = lines[block.start - 1]
        indent = first[: len(first) - len(first.lstrip())]
        lines[block.end : block.end] = [f"{indent}{MARK_END}\n"]
        lines[block.start - 1 : block.start - 1] = [f"{indent}{MARK_START}\n"]
    return "".join(lines)
