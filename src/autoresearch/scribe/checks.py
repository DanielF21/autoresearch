"""Deterministic gates on a draft. A draft that fails one is never shown to a judge.

Number tracing is the strict one: every number in the title and body must match a
value the record holds, in one of a few stated renderings. Everything a judge
could be fooled by, a number cannot fake. It starts strict on purpose; a false
positive is fixed by adding a rendering, never by skipping the check.

Lint patterns are method data. A pattern this repository's own merged bodies use
more often than the method allows is dropped for this corpus, so the lint encodes
what reads as machine written here, not a taste imposed from outside.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from autoresearch.scribe.corpus_stats import body_stats, pattern_rate, title_prefix
from autoresearch.scribe.facts import Facts, parse_diff
from autoresearch.scribe.method import WriterMethod
from autoresearch.scribe.runread import Candidate

_URL = re.compile(r"\[([^\]]*)\]\([^)]*\)|https?://\S+")
_LIST_MARKER = re.compile(r"^(\s*)\d+[.)](\s)", re.M)
_REF = re.compile(r"(?<![\w&])#(\d+)")
_NUMBER = re.compile(
    r"(?<![\w.#@/^\-])"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>x|"
    + "\N{MULTIPLICATION SIGN}"
    + r"|%|ms|µs|us|s|sec|seconds?|milliseconds?|times)(?![\w]))?"
    r"(?![\w.]\d|\w)"
)
ALWAYS_ALLOWED = (0.0, 1.0)
RATIO_UNITS = ("x", "\N{MULTIPLICATION SIGN}", "times")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": list(self.detail)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckResult:
        return cls(str(d["name"]), bool(d["ok"]), tuple(str(x) for x in d.get("detail", [])))


@dataclass(frozen=True)
class FactSheet:
    ratios: tuple[float, ...]
    floors: tuple[float, ...]
    seconds: tuple[float, ...]
    counts: tuple[float, ...]
    literals: tuple[float, ...]
    refs: tuple[int, ...]
    inputs: tuple[tuple[str, str], ...]


def _numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?", text):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def build_fact_sheet(c: Candidate, facts: Facts, refs: tuple[int, ...] = ()) -> FactSheet:
    ratios = [r for r in (c.speedup, c.worst_speedup) if r is not None]
    seconds: list[float] = []
    counts: list[float] = [float(len(c.inputs)), float(len(facts.files)), float(facts.hunks)]
    counts += [float(facts.added), float(facts.removed), float(facts.changed_lines)]
    for i in c.inputs:
        if i.speedup is not None:
            ratios.append(i.speedup)
        seconds += [s for s in (i.base_median_s, i.patched_median_s) if s is not None]
        counts += [float(i.pairs), float(i.clean_pairs)]
    for t in c.tests:
        counts += [float(t.passed), float(t.failed), float(t.errors)]
    literals: list[float] = []
    for fd in parse_diff(c.patch or ""):
        for line in [*fd.added.values(), *fd.removed.values()]:
            literals += _numbers_in(line)
    for i in c.inputs:
        literals += _numbers_in(i.setup)
    return FactSheet(
        ratios=tuple(ratios),
        floors=tuple(i.noise_floor for i in c.inputs),
        seconds=tuple(seconds),
        counts=tuple(counts),
        literals=tuple(literals),
        refs=refs,
        inputs=tuple((i.name, i.setup) for i in c.inputs),
    )


def _scrub(text: str) -> str:
    text = _URL.sub(lambda m: m.group(1) or "", text)
    return _LIST_MARKER.sub(r"\1\2", text)


def _close(value: float, decimals: int, target: float) -> bool:
    return abs(target - value) <= 0.5 * 10.0**-decimals + 1e-9


@dataclass(frozen=True)
class NumberToken:
    text: str
    value: float
    decimals: int
    unit: str
    start: int
    end: int


def extract_numbers(text: str) -> list[NumberToken]:
    scrubbed = _scrub(text)
    out: list[NumberToken] = []
    for m in _NUMBER.finditer(scrubbed):
        raw = m.group("num")
        digits = raw.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        unit = (m.group("unit") or "").lower()
        out.append(NumberToken(m.group(0), float(digits), decimals, unit, m.start(), m.end()))
    return out


def _targets(token: NumberToken, sheet: FactSheet) -> list[float]:
    u = token.unit
    if u in RATIO_UNITS:
        return list(sheet.ratios)
    if u == "%":
        return (
            [(r - 1) * 100 for r in sheet.ratios]
            + [(1 - 1 / r) * 100 for r in sheet.ratios if r]
            + [(f - 1) * 100 for f in sheet.floors]
        )
    if u in ("ms", "millisecond", "milliseconds"):
        return [s * 1000 for s in sheet.seconds]
    if u in ("µs", "us"):
        return [s * 1e6 for s in sheet.seconds]
    if u in ("s", "sec", "second", "seconds"):
        return list(sheet.seconds)
    return [
        *ALWAYS_ALLOWED,
        *sheet.ratios,
        *sheet.floors,
        *sheet.counts,
        *sheet.literals,
        *sheet.seconds,
    ]


def trace_numbers(title: str, body: str, sheet: FactSheet) -> CheckResult:
    text = f"{title}\n{body}"
    stray: list[str] = []
    for token in extract_numbers(text):
        if not any(_close(token.value, token.decimals, t) for t in _targets(token, sheet)):
            stray.append(token.text.strip())
    for ref in _REF.findall(_scrub(text)):
        if int(ref) not in sheet.refs:
            stray.append(f"#{ref}")
    return CheckResult(
        "numbers trace to the record",
        not stray,
        tuple(f"{s} matches no measured or recorded value" for s in stray),
    )


def check_direction(title: str, body: str, sheet: FactSheet) -> CheckResult:
    text = _scrub(f"{title}\n{body}")
    wrong: list[str] = []
    for token in extract_numbers(text):
        if token.unit not in (*RATIO_UNITS, "%"):
            continue
        if token.unit == "%":
            matched = [
                r
                for r in sheet.ratios
                if _close(token.value, token.decimals, (r - 1) * 100)
                or (r and _close(token.value, token.decimals, (1 - 1 / r) * 100))
            ]
        else:
            matched = [r for r in sheet.ratios if _close(token.value, token.decimals, r)]
        if not matched:
            continue
        window = text[token.end : token.end + 40].lower()
        if "slower" in window and all(r > 1 for r in matched):
            wrong.append(f"{token.text.strip()} called slower, but the measurement is faster")
        if "faster" in window and all(r < 1 for r in matched):
            wrong.append(f"{token.text.strip()} called faster, but the measurement is slower")
    return CheckResult("faster and slower agree with the measurement", not wrong, tuple(wrong))


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def check_coverage(body: str, sheet: FactSheet) -> CheckResult:
    """Every measured input is named, or its setup expression appears."""
    flat = _squash(body)
    missing: list[str] = []
    for name, setup in sheet.inputs:
        rhs = setup.split("=", 1)[1] if "=" in setup else setup
        rhs_flat = _squash(rhs)
        keys = {name.lower(), rhs_flat}
        head = ",".join(rhs_flat.split(",")[:2])
        keys.add(head)
        keys.add(head.split(".", 1)[1] if "." in head.split("(")[0] else head)
        if not any(k and k in flat for k in keys):
            missing.append(name)
    return CheckResult(
        "every measured input appears",
        not missing,
        tuple(f"{n} is not mentioned by name or setup" for n in missing),
    )


def lint(
    title: str, body: str, method: WriterMethod, corpus_bodies: list[str]
) -> tuple[CheckResult, tuple[str, ...]]:
    """The lint result, and the patterns dropped because this corpus uses them."""
    text = f"{title}\n{body}"
    hits: list[str] = []
    dropped: list[str] = []
    for p in method.slop:
        if pattern_rate(p.regex, corpus_bodies) > method.lint_max_corpus_rate:
            dropped.append(p.name)
            continue
        found = [m.group(0) for m in p.regex.finditer(text)]
        if found:
            hits.append(f"{p.name}: {', '.join(repr(f) for f in found[:5])}")
    return CheckResult("no slop patterns", not hits, tuple(hits)), tuple(dropped)


def check_structure(title: str, body: str, stats: dict[str, Any]) -> CheckResult:
    if not stats:
        return CheckResult("structure inside the corpus range", False, ("corpus has no stats",))
    problems: list[str] = []
    st = body_stats(body)
    ranges = stats["body"]
    lo, hi = ranges["chars"]["min"], ranges["chars"]["max"]
    if not lo <= st.chars <= hi:
        problems.append(
            f"body is {st.chars} characters; merged bodies here run {lo:.0f} to {hi:.0f}"
        )
    for field in ("headers", "bullet_lines", "code_fences", "numbered_lines"):
        mx = ranges[field]["max"]
        if getattr(st, field) > mx:
            problems.append(f"{field.replace('_', ' ')}: {getattr(st, field)}, corpus max {mx:.0f}")
    prefix = title_prefix(title)
    if prefix not in stats["prefixes"]:
        allowed = ", ".join(repr(p) for p in sorted(stats["prefixes"])) or "none"
        problems.append(f"title prefix {prefix!r} is not used here; used: {allowed}")
    if len(title) > stats["title_chars"]["max"]:
        problems.append(
            f"title is {len(title)} characters; corpus max {stats['title_chars']['max']:.0f}"
        )
    if "<!--" in body:
        problems.append("template comments were left in the body")
    return CheckResult("structure inside the corpus range", not problems, tuple(problems))


def run_checks(
    title: str,
    body: str,
    sheet: FactSheet,
    method: WriterMethod,
    stats: dict[str, Any],
    corpus_bodies: list[str],
) -> tuple[CheckResult, ...]:
    results = [
        trace_numbers(title, body, sheet),
        check_direction(title, body, sheet),
        lint(title, body, method, corpus_bodies)[0],
        check_structure(title, body, stats),
    ]
    if method.require_input_coverage:
        results.append(check_coverage(body, sheet))
    return tuple(results)


def passed(results: tuple[CheckResult, ...]) -> bool:
    return bool(results) and all(r.ok for r in results)
