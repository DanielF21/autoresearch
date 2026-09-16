"""The block rule, and applying SEARCH/REPLACE edits to blocks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from alphaevolve.blocks import (
    MARK_END,
    MARK_START,
    Block,
    BlocksError,
    EditError,
    apply_edits,
    assemble,
    candidate_paths,
    derive,
    marked,
    parse_edits,
    strip_edits,
)
from autoresearch.config import load_config
from tests.ae_helpers import DIVIDER, REPLACE, ROOT, SEARCH, edit_reply

SOURCE = """import functools


def helper(x):
    return x + 1


class Parser:
    @functools.lru_cache
    def parse(self, text):
        out = []
        for ch in text:
            out.append(helper(ord(ch)))
        return out

    def small(self):
        def inner():
            return 1
        return inner()
"""

FLAT = """   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     5000    0.300    0.000    0.500    0.000 pkg/mod.py:9(parse)
  40000/2    0.200    0.000    0.200    0.000 pkg/mod.py:4(helper)
      100    0.050    0.000    0.050    0.000 pkg/mod.py:17(inner)
      100    0.040    0.000    0.090    0.000 pkg/mod.py:16(small)
    70228    0.008    0.000    0.008    0.000 {built-in method builtins.len}
       10    0.001    0.000    0.001    0.000 tests/test_mod.py:3(test_x)
"""
ALLOW = ("pkg/**",)
DENY = ("tests/**",)


def _blocks(cap: int = 10_000) -> tuple[Block, ...]:
    return derive(FLAT, {"pkg/mod.py": SOURCE}, ALLOW, DENY, cap)


def test_blocks_follow_the_profile_skip_overlaps_and_sit_in_line_order() -> None:
    assert candidate_paths(FLAT, ALLOW, DENY) == ("pkg/mod.py",)
    blocks = _blocks()
    assert [(b.function, b.start, b.end) for b in blocks] == [
        ("helper", 4, 5),
        ("parse", 9, 14),  # the decorator line is where cProfile names it
        ("inner", 17, 18),  # small encloses inner, which ranked first, so small is skipped
    ]
    lines = SOURCE.splitlines(keepends=True)
    for b in blocks:
        assert b.text == "".join(lines[b.start - 1 : b.end])
        assert Block.from_dict(b.to_dict()) == b


def test_the_cap_stops_before_the_block_that_exceeds_it_but_keeps_the_first() -> None:
    parse_len = len(_blocks()[1].text)
    assert [b.function for b in _blocks(parse_len)] == ["parse"]
    assert [b.function for b in _blocks(1)] == ["parse"]


def test_no_function_in_scope_is_an_error() -> None:
    with pytest.raises(BlocksError):
        derive(FLAT, {"pkg/mod.py": SOURCE}, ("other/**",), DENY, 10_000)


def test_edits_apply_to_the_one_place_they_match() -> None:
    blocks = _blocks()
    texts = tuple(b.text for b in blocks)
    reply = edit_reply("        return out", "        return list(out)") + edit_reply(
        "    return x + 1", "    return 1 + x", prose="And this."
    )
    edits = parse_edits(reply)
    assert edits == (
        ("        return out", "        return list(out)"),
        ("    return x + 1", "    return 1 + x"),
    )
    new = apply_edits(texts, edits)
    assert new[0] == "def helper(x):\n    return 1 + x\n"
    assert new[1].endswith("        return list(out)\n")
    assert new[2] == texts[2]
    assert strip_edits(reply) == "Faster.\n[edit]\nAnd this.\n[edit]"


def test_an_edit_that_matches_nowhere_or_is_empty_is_refused() -> None:
    texts = tuple(b.text for b in _blocks())
    with pytest.raises(EditError, match="no SEARCH/REPLACE"):
        apply_edits(texts, parse_edits("no edits here"))
    with pytest.raises(EditError, match="matches no marked block"):
        apply_edits(texts, [("    return small", "x")])
    with pytest.raises(EditError, match="empty"):
        apply_edits(texts, [("   ", "x")])


def test_a_search_matching_in_two_blocks_is_ambiguous() -> None:
    texts = ("a = 1\nb = 2\n", "b = 2\n")
    with pytest.raises(EditError, match="2 places"):
        apply_edits(texts, [("b = 2", "b = 3")])


def test_an_empty_replacement_deletes_and_texts_keep_a_final_newline() -> None:
    new = apply_edits(("a = 1\nb = 2\n",), [("b = 2", "")])
    assert new == ("a = 1\n",)
    new = apply_edits(("a = 1\n",), [("a = 1", "")])
    assert new == ("\n",)


def test_assemble_and_markers() -> None:
    blocks = _blocks()
    texts = [b.text for b in blocks]
    assert assemble({"pkg/mod.py": SOURCE}, blocks, texts) == {"pkg/mod.py": SOURCE}
    texts[1] = "    def parse(self, text):\n        return list(text)\n"
    files = assemble({"pkg/mod.py": SOURCE}, blocks, texts)
    assert "@functools.lru_cache" not in files["pkg/mod.py"]
    assert "def helper(x):" in files["pkg/mod.py"] and "def small(self):" in files["pkg/mod.py"]
    shown = marked(SOURCE, blocks).splitlines()
    assert shown[3:7] == [MARK_START, "def helper(x):", "    return x + 1", MARK_END]
    assert shown[10] == f"    {MARK_START}" and shown[11] == "    @functools.lru_cache"
    assert [line.strip() for line in shown].count(MARK_START) == 3


def test_markers_are_built_not_written() -> None:
    assert SEARCH.endswith(" SEARCH") and len(DIVIDER) == 7 and REPLACE.endswith(" REPLACE")


def _show(clone: Path, sha: str, path: str) -> str | None:
    r = subprocess.run(
        ["git", "-C", str(clone), "show", f"{sha}:{path}"], capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else None


@pytest.mark.parametrize(
    ("config", "doc", "clone"),
    [
        (
            "configs/seeded/pyparsing_w16.toml",
            "configs/seeded/docs/pyparsing_w16/profile_flat_long_flat.txt",
            "runs/intake_s/pyparsing/repo",
        ),
        (
            "configs/seeded/pycodestyle_w16.toml",
            "configs/seeded/docs/pycodestyle_w16/profile_flat_assignments_flat.txt",
            "runs/intake/pycodestyle/repo",
        ),
    ],
)
def test_the_rule_on_the_real_profiles_and_sources(config: str, doc: str, clone: str) -> None:
    target = load_config(ROOT / config).target
    flat = (ROOT / doc).read_text()
    paths = candidate_paths(flat, target.allow, target.deny)
    sources = {p: _show(ROOT / clone, target.sha, p) for p in paths}
    if not (ROOT / clone).exists() or any(s is None for s in sources.values()):
        pytest.skip(f"no local clone of {target.name} at {target.sha[:12]}")
    present = {p: s for p, s in sources.items() if s is not None}
    blocks = derive(flat, present, target.allow, target.deny, 10_000)
    assert blocks
    assert sum(len(b.text) for b in blocks) <= 10_000 or len(blocks) == 1
    for b in blocks:
        lines = present[b.path].splitlines(keepends=True)
        assert b.text == "".join(lines[b.start - 1 : b.end])
        assert b.text.lstrip().startswith(("def ", "async def ", "@"))
    texts = [b.text for b in blocks]
    assert assemble(present, blocks, texts) == {b.path: present[b.path] for b in blocks}
