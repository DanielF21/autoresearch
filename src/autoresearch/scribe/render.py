"""Text a model is shown about a candidate, rendered by code from measured records.

Numbers are printed here once, in one format, so the number tracing check and
every prompt agree on what was measured.
"""

from __future__ import annotations

from autoresearch.scribe.facts import Facts
from autoresearch.scribe.runread import Candidate


def fmt_ratio(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}x"


def fmt_seconds(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 1000:.1f} ms" if v < 1 else f"{v:.3f} s"


def inputs_table(c: Candidate) -> str:
    rows = [
        "| input | setup | speedup | noise floor | clean pairs | base median | patched median | same result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in c.inputs:
        same = {True: "yes", False: "NO", None: "unknown"}[i.result_matches]
        rows.append(
            f"| {i.name} | `{i.setup}` | {fmt_ratio(i.speedup)} | {i.noise_floor:.4f} | "
            f"{i.clean_pairs} of {i.pairs} | {fmt_seconds(i.base_median_s)} | "
            f"{fmt_seconds(i.patched_median_s)} | {same} |"
        )
    rows.append("")
    rows.append(
        f"Geometric mean over {len(c.inputs)} inputs: {fmt_ratio(c.speedup)}. "
        f"Worst input: {fmt_ratio(c.worst_speedup)}. "
        "A speedup is base seconds over patched seconds, the median of the clean pairs."
    )
    return "\n".join(rows)


def tests_line(c: Candidate) -> str:
    if not c.tests:
        return "No test results recorded."
    parts = [
        f"{t.scope} suite: {t.passed} passed, {t.failed} failed, {t.errors} errors" for t in c.tests
    ]
    return "; ".join(parts) + "."


def facts_block(f: Facts) -> str:
    lines = [
        f"Files: {', '.join(f.files) or 'none'}",
        f"Lines: {f.added} added, {f.removed} removed, {f.hunks} hunks",
        "Functions touched: "
        + (
            ", ".join(
                f"{fc.path}::{fc.name} (+{fc.added} -{fc.removed}"
                + (", old path kept" if fc.old_path_kept else "")
                + ")"
                for fc in f.functions
            )
            or "none"
        ),
        f"New imports: {', '.join(f.new_imports) or 'none'}",
        f"Public API changes: {'; '.join(f.public_api_changes) or 'none'}",
        "New numeric literals in comparisons: " + ("; ".join(f.new_condition_numbers) or "none"),
        "Private attribute reads on objects other than self: "
        + ("; ".join(f.private_attribute_reads) or "none"),
    ]
    if f.parse_errors:
        lines.append(f"Parse errors: {'; '.join(f.parse_errors)}")
    return "\n".join(f"- {ln}" for ln in lines)


def outcome(c: Candidate) -> str:
    """One line on what happened to an attempt, from the record alone."""
    if not c.patch:
        return f"no patch ({c.stop_reason}{': ' + c.error[:120] if c.error else ''})"
    if not c.measured:
        return "patch not measured"
    if not c.applied:
        return "patch did not apply"
    if not c.tests_pass:
        return "tests failed"
    if c.result_matches is not True:
        return "results differ from base"
    if c.regressions:
        slow = ", ".join(
            f"{i.name} {fmt_ratio(i.speedup)}" for i in c.inputs if i.name in c.regressions
        )
        return f"slower on {slow}; geomean {fmt_ratio(c.speedup)}"
    if not c.clears_noise:
        return f"inside the noise floor; geomean {fmt_ratio(c.speedup)}"
    dup = f", duplicate of {c.duplicate_of}" if c.duplicate_of else ""
    return f"clears noise, geomean {fmt_ratio(c.speedup)}, worst {fmt_ratio(c.worst_speedup)}{dup}"
