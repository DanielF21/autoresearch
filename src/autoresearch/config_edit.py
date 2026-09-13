"""Edits to a config's text that a measurement writes back: noise floors and docs.

A config is edited as text, never re-rendered, so every comment and every line
nobody asked to change stays byte for byte. Each edit is checked by parsing the
result: the new config must equal the old one with only the named fields
changed, or nothing is returned.

The scanner tracks TOML string state across lines, so a ``[`` or a
``name =`` inside a multiline ``setup`` is never mistaken for structure.

Never point this at a config frozen inside a run directory. A run's config is
its bytes; ``init_run`` refuses to resume one whose text changed.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal

from autoresearch.config import RunConfig, parse_config

INPUTS_HEADER = re.compile(r"^\s*\[\[\s*target\.inputs\s*\]\]")
ANY_HEADER = re.compile(r"^\s*\[")
TARGET_HEADER = re.compile(r"^\s*\[\s*target\s*\]")


class ConfigEditError(ValueError):
    """An edit that could not be made, or that would change more than it names."""


@dataclass(frozen=True)
class _Line:
    text: str
    starts_in_string: bool
    ends_in_string: bool


def _scan(text: str) -> list[_Line]:
    """Each line, with whether it begins and ends inside a multiline string."""
    out: list[_Line] = []
    state = ""  # "", '"""' or "'''"
    for line in text.split("\n"):
        start = state
        i = 0
        while i < len(line):
            if state:
                if state == '"""' and line[i] == "\\":
                    i += 2
                    continue
                if line.startswith(state, i):
                    i += 3
                    state = ""
                    continue
                i += 1
                continue
            ch = line[i]
            if ch == "#":
                break
            if line.startswith('"""', i) or line.startswith("'''", i):
                state = line[i : i + 3]
                i += 3
                continue
            if ch in "\"'":
                end = i + 1
                while end < len(line) and line[end] != ch:
                    end += 2 if ch == '"' and line[end] == "\\" else 1
                i = end + 1
                continue
            i += 1
        out.append(_Line(line, bool(start), bool(state)))
    return out


def _structural(line: _Line) -> bool:
    return not line.starts_in_string


def _is_header(line: _Line) -> bool:
    return _structural(line) and bool(ANY_HEADER.match(line.text))


def _key_value(lines: list[_Line], i: int) -> tuple[str, object, int] | None:
    """The key on line ``i`` and its value, plus the index after its last line."""
    if not _structural(lines[i]) or _is_header(lines[i]):
        return None
    body = lines[i].text.strip()
    if not body or body.startswith("#") or "=" not in body:
        return None
    chunk = lines[i].text
    j = i
    while True:
        try:
            parsed = tomllib.loads(chunk)
        except tomllib.TOMLDecodeError:
            j += 1
            if j >= len(lines) or _is_header(lines[j]):
                return None
            chunk += "\n" + lines[j].text
            continue
        if len(parsed) != 1:
            return None
        key, value = next(iter(parsed.items()))
        return key, value, j + 1


def _table_end(lines: list[_Line], start: int) -> int:
    end = start + 1
    while end < len(lines) and not _is_header(lines[end]):
        end += 1
    return end


def _insert_at(lines: list[_Line], start: int, end: int) -> int:
    """Just after the last line of the table that belongs to a key."""
    last = start + 1
    i = start + 1
    while i < end:
        kv = _key_value(lines, i)
        if kv is None:
            i += 1
            continue
        last = kv[2]
        i = kv[2]
    return last


def _check(new_text: str, expected: RunConfig) -> str:
    try:
        new = parse_config(new_text)
    except ValueError as e:
        raise ConfigEditError(f"the edited config does not load: {e}") from e
    if replace(new, source_text="") != replace(expected, source_text=""):
        raise ConfigEditError("the edit changed more than the fields it names; nothing written")
    return new_text


def set_noise_floors(text: str, floors: Mapping[str, tuple[float, str]]) -> str:
    """``text`` with each named input's ``noise_floor`` set, rounded up to four decimals.

    Up, so the config never sits under the floor that was measured.

    ``floors`` maps an input name to its floor and a comment for the line. An
    existing ``noise_floor`` line in that input's table is replaced, comment and
    all; otherwise the line goes directly after ``name``.
    """
    old = parse_config(text)
    names = {i.name for i in old.target.inputs}
    unknown = sorted(set(floors) - names)
    if unknown:
        raise ConfigEditError(f"no such input: {', '.join(unknown)}")

    lines = _scan(text)
    out = [ln.text for ln in lines]
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement) over line indices
    written: dict[str, float] = {}
    i = 0
    while i < len(lines):
        if not (_structural(lines[i]) and INPUTS_HEADER.match(lines[i].text)):
            i += 1
            continue
        end = _table_end(lines, i)
        name_at: tuple[int, str] | None = None
        floor_at: tuple[int, int] | None = None
        j = i + 1
        while j < end:
            kv = _key_value(lines, j)
            if kv is None:
                j += 1
                continue
            key, value, after = kv
            if key == "name" and isinstance(value, str):
                name_at = (after, value)
            elif key == "noise_floor":
                floor_at = (j, after)
            j = after
        if name_at is not None and name_at[1] in floors:
            floor, comment = floors[name_at[1]]
            rendered = _four_places_up(floor)
            line = f"noise_floor = {rendered}" + (f"   # {comment}" if comment else "")
            if floor_at is not None:
                edits.append((floor_at[0], floor_at[1], line))
            else:
                edits.append((name_at[0], name_at[0], line))
            written[name_at[1]] = float(rendered)
        i = end

    missing = sorted(set(floors) - set(written))
    if missing:
        raise ConfigEditError(f"no [[target.inputs]] table found for: {', '.join(missing)}")
    for start, end, line in sorted(edits, reverse=True):
        out[start:end] = [line]

    inputs = tuple(
        replace(i, noise_floor=written[i.name]) if i.name in written else i
        for i in old.target.inputs
    )
    expected = replace(old, target=replace(old.target, inputs=inputs))
    return _check("\n".join(out), expected)


def set_docs(text: str, docs: Sequence[str]) -> str:
    """``text`` with ``[target].docs`` set to ``docs``, replacing any existing list."""
    old = parse_config(text)
    lines = _scan(text)
    out = [ln.text for ln in lines]
    start = next(
        (i for i, ln in enumerate(lines) if _structural(ln) and TARGET_HEADER.match(ln.text)),
        None,
    )
    if start is None:
        raise ConfigEditError("no [target] table")
    end = _table_end(lines, start)
    rendered = "docs = [" + ", ".join(_basic_string(d) for d in docs) + "]"
    i = start + 1
    while i < end:
        kv = _key_value(lines, i)
        if kv is None:
            i += 1
            continue
        key, _value, after = kv
        if key == "docs":
            out[i:after] = [rendered]
            break
        i = after
    else:
        at = _insert_at(lines, start, end)
        out[at:at] = [rendered]
    expected = replace(old, target=replace(old.target, docs=tuple(docs)))
    return _check("\n".join(out), expected)


def _four_places_up(value: float) -> str:
    return str(Decimal(repr(value)).quantize(Decimal("0.0001"), rounding=ROUND_CEILING))


def _basic_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
