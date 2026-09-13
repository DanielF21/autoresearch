"""What the writer is given, built by code and saved exactly as sent.

The dossier is the record, not a summary of it: the diff, the facts, the measured
table, the test counts, the proposing agent's own notes, and one line per other
attempt in the search with what happened to it. The other attempts are where a PR
body finds something the diff cannot show, so each one that touched the same file
is flagged. The full artifacts stay readable through the ``run:`` root.
"""

from __future__ import annotations

from autoresearch.scribe.corpus import CorpusPR, render_pr
from autoresearch.scribe.facts import Facts
from autoresearch.scribe.render import facts_block, fmt_ratio, inputs_table, outcome, tests_line
from autoresearch.scribe.runread import Candidate, RunTarget

RATIONALE_CLIP = 600


def _first_paragraph(text: str, limit: int = RATIONALE_CLIP) -> str:
    para = text.strip().split("\n\n")[0].strip().replace("\n", " ")
    return para if len(para) <= limit else para[:limit] + " ..."


def search_log(facts: Facts, others: list[tuple[Candidate, Facts | None]]) -> str:
    lines = [
        "One line per other attempt in this search. Each was proposed from the same base "
        "commit and measured the same way. Full patches, measurements and notes are under "
        "run:attempts/. Notes are the proposing agent's own words, not verified.",
        "",
    ]
    mine = set(facts.files)
    for o, of in sorted(others, key=lambda x: x[0].number):
        same = bool(of and mine & set(of.files))
        tag = " [same file]" if same else ""
        predicted = f"; predicted {fmt_ratio(o.predicted_speedup)}" if o.predicted_speedup else ""
        lines.append(f"- attempt {o.number}{tag}: {outcome(o)}{predicted}")
        if o.rationale.strip():
            lines.append(f"  notes: {_first_paragraph(o.rationale)}")
    if len(lines) == 2:
        lines.append("(no other attempts)")
    return "\n".join(lines)


def build_dossier(
    c: Candidate,
    facts: Facts,
    target: RunTarget,
    others: list[tuple[Candidate, Facts | None]],
    template: str,
) -> str:
    """Everything but the exemplars, which vary per draft and are appended by the writer step."""
    parts = [
        f"# The change: attempt {c.number} of search {target.run_id}",
        "",
        f"Repository: {target.repo}",
        f"Base commit the patch was written and measured against: {target.sha}",
        "",
        "## Diff",
        "```diff",
        (c.patch or "").rstrip(),
        "```",
        "",
        "## Facts computed from the diff and the syntax trees",
        facts_block(facts),
        "",
        "## Measurements",
        "Each input was timed in alternating pairs of base and patched runs on one "
        "machine. Each input's result was fingerprinted on both trees and compared.",
        "",
        inputs_table(c),
        "",
        "## Tests",
        tests_line(c),
        "",
        "## The proposing agent's notes",
        f"Predicted speedup before measurement: {fmt_ratio(c.predicted_speedup)}",
        "",
        c.rationale.strip() or "(none)",
        "",
        "## Other attempts in the search",
        search_log(facts, others),
    ]
    if template.strip():
        parts += [
            "",
            "## This repository's pull request template",
            "```",
            template.strip(),
            "```",
        ]
    return "\n".join(parts) + "\n"


def exemplar_block(exemplars: list[CorpusPR]) -> str:
    parts = [
        "## Pull requests merged in this repository",
        "Real titles and descriptions, for how contributors here write. Do not copy their content.",
    ]
    for pr in exemplars:
        parts += ["", "---", "", render_pr(pr)]
    return "\n".join(parts)


def rotate(prs: tuple[CorpusPR, ...], count: int, offset: int) -> list[CorpusPR]:
    if not prs:
        return []
    count = min(count, len(prs))
    return [prs[(offset + i) % len(prs)] for i in range(count)]
