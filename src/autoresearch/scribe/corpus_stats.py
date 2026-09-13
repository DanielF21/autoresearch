"""Style statistics of a corpus, computed in code.

The structure gate compares a draft against these ranges rather than against a
notion of good style written into a prompt. With 10 to 20 bodies, quantiles are
too coarse to mean anything, so only min, median and max are kept.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from autoresearch.scribe.corpus import CorpusPR

_HEADER = re.compile(r"^#{1,6}\s")
_BULLET = re.compile(r"^\s*[-*+]\s")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s")
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)|https?://\S+")
_REF = re.compile(r"(?<![\w&])#\d+")
_INLINE_CODE = re.compile(r"`[^`\n]+`")


@dataclass(frozen=True)
class BodyStats:
    chars: int
    words: int
    lines: int
    paragraphs: int
    headers: int
    bullet_lines: int
    numbered_lines: int
    code_fences: int
    inline_code: int
    links: int
    refs: int


def body_stats(body: str) -> BodyStats:
    lines = body.splitlines()
    in_fence = False
    headers = bullets = numbered = fences = 0
    for line in lines:
        if line.strip().startswith("```"):
            fences += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        headers += bool(_HEADER.match(line))
        bullets += bool(_BULLET.match(line))
        numbered += bool(_NUMBERED.match(line))
    paragraphs = len([p for p in re.split(r"\n\s*\n", body.strip()) if p.strip()])
    return BodyStats(
        chars=len(body.strip()),
        words=len(body.split()),
        lines=len([ln for ln in lines if ln.strip()]),
        paragraphs=paragraphs,
        headers=headers,
        bullet_lines=bullets,
        numbered_lines=numbered,
        code_fences=fences // 2,
        inline_code=len(_INLINE_CODE.findall(body)),
        links=len(_LINK.findall(body)),
        refs=len(_REF.findall(body)),
    )


def title_prefix(title: str) -> str:
    """``BUG: x`` is BUG, ``[BUG][PERF] x`` is BUG+PERF, and no prefix is the empty string."""
    m = re.match(r"^((?:\[[A-Za-z]{2,}\])+)\s", title)
    if m:
        return "+".join(t.upper() for t in re.findall(r"\[([A-Za-z]+)\]", m.group(1)))
    m = re.match(r"^([A-Za-z]{2,}):\s", title)
    return m.group(1).upper() if m else ""


def _summary(values: Iterable[float]) -> dict[str, float]:
    vals = list(values)
    if not vals:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {"min": min(vals), "median": float(statistics.median(vals)), "max": max(vals)}


def corpus_stats(prs: list[CorpusPR]) -> dict[str, Any]:
    per = [body_stats(pr.body) for pr in prs]
    fields = list(asdict(per[0]).keys()) if per else list(BodyStats.__dataclass_fields__)
    by_prefix: dict[str, list[BodyStats]] = {}
    for pr, st in zip(prs, per, strict=True):
        by_prefix.setdefault(title_prefix(pr.title), []).append(st)
    return {
        "n": len(prs),
        "body": {f: _summary(getattr(s, f) for s in per) for f in fields},
        "title_chars": _summary(len(pr.title) for pr in prs),
        "prefixes": dict(Counter(title_prefix(pr.title) for pr in prs)),
        "by_prefix": {
            prefix: {"n": len(group), "chars": _summary(s.chars for s in group)}
            for prefix, group in sorted(by_prefix.items())
        },
    }


def pattern_rate(pattern: re.Pattern[str], texts: list[str]) -> float:
    """Share of texts in which the pattern fires at least once."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if pattern.search(t)) / len(texts)
