"""The two MAP Elites feature values, ported from OpenEvolve.

Source: codelion/openevolve at 411fb59c886c18704caaffb611e17cf9e7d824d2,
openevolve/database.py: ``complexity`` is ``len(program.code)``
(``_calculate_feature_coords``) and ``diversity`` is ``_fast_code_diversity``,
kept with its constants. A program's code here is its block texts joined.

The shape of a code string is cached: the database compares every new program
against a reference set, and rebuilding it from a run directory compares
thousands, each of whose character set is the same every time.
"""

from __future__ import annotations

from functools import lru_cache


def complexity(code: str) -> int:
    return len(code)


@lru_cache(maxsize=8192)
def _shape(code: str) -> tuple[int, int, frozenset[str]]:
    return len(code), code.count("\n"), frozenset(code)


def fast_code_diversity(code1: str, code2: str) -> float:
    """Higher is more diverse. OpenEvolve's constants: 0.1 per character of length
    difference, 10 per line of line count difference, 0.5 per character in only
    one of the two character sets."""
    if code1 == code2:
        return 0.0
    len1, lines1, chars1 = _shape(code1)
    len2, lines2, chars2 = _shape(code2)
    return abs(len1 - len2) * 0.1 + abs(lines1 - lines2) * 10 + len(chars1 ^ chars2) * 0.5
