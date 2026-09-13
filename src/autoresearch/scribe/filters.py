"""Hard filters, in code. A candidate a filter drops is never shown to a model.

Two bars. Reviewable: the patch was measured and every fact the referee records
is clean. Writable: reviewable, and timed on at least ``min_inputs`` inputs,
because a single input record cannot show how a patch behaves across input shapes,
which is the first thing a maintainer asks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoresearch.scribe.runread import Candidate


@dataclass(frozen=True)
class FilterResult:
    number: int
    reviewable: bool
    writable: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "reviewable": self.reviewable,
            "writable": self.writable,
            "reasons": list(self.reasons),
        }


def filter_candidate(c: Candidate, min_inputs: int) -> FilterResult:
    reasons: list[str] = []
    if not c.patch:
        reasons.append(f"no patch ({c.skipped or c.stop_reason})")
    elif not c.measured:
        reasons.append("not measured")
    else:
        if not c.applied:
            reasons.append("did not apply")
        if c.scope_violations:
            reasons.append("scope violations: " + ", ".join(c.scope_violations))
        if not c.tests_pass:
            reasons.append("tests did not pass")
        if c.result_matches is not True:
            reasons.append("results do not match on every input")
        if c.regressions:
            reasons.append("regresses on: " + ", ".join(c.regressions))
        if not c.clears_noise:
            reasons.append("does not clear the noise floor")
    if c.duplicate_of:
        reasons.append(f"duplicate of attempt {c.duplicate_of}")
    reviewable = not reasons
    writable = reviewable and len(c.inputs) >= min_inputs
    if reviewable and not writable:
        reasons.append(f"timed on {len(c.inputs)} input(s), fewer than {min_inputs}")
    return FilterResult(c.number, reviewable, writable, tuple(reasons))
