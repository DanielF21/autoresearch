"""Whether a target can be measured by this harness, judged from a survey.

The rules follow from how the referee measures, not from any target:

1. The hot file must execute on every input. The referee attributes cProfile
   self time to it and refuses to time an input on which it did not run.
2. The result must be deterministic: two launches of the base tree must
   fingerprint the same, or ``result_matches`` means nothing.
3. A call must sit inside the band one launch can time: long enough that a
   calibrated floor is tighter than a few percent, short enough that
   ``repeats_per_launch`` calls fit inside ``LAUNCH_TIMEOUT``.
4. Both suites must pass on the base tree and finish inside their timeout.

Two things are warnings rather than failures, because they change what a run
means without stopping it: a hot file with a small share of self time (the
agents will be optimising a loop around something else, most often a C
call), and a module suite that is the whole suite (the referee runs it twice
per attempt). Uncalibrated inputs are listed, not judged: calibration comes
after admission.

Threading is not checked. Every timing launch is pinned to one core, so a
call that spreads work across threads is measured serialised and a patch that
adds parallelism measures as no faster. That is a property of the design,
stated in the README, and nothing in a survey can see it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from autoresearch.config import TargetSpec
from autoresearch.referee.referee import LAUNCH_TIMEOUT, TESTS_TIMEOUT, Survey

CALL_MIN_S = 0.005
# Headroom under the launch timeout for interpreter start, import and setup.
LAUNCH_HEADROOM_S = 40.0
HOT_SHARE_WARN = 0.2
# Chosen, not measured: seconds of timing launches one attempt may cost the referee.
# The estimate uses call times under cProfile, which run slower than plain calls.
# Networkx's five inputs take about 2.3 s a call plainly (t1_w4d base medians), about
# 200 s per attempt, so they fit even at several times that under cProfile.
# Calibration times seven attempts' worth of pairs, so this also bounds it.
REFEREE_TIMING_BUDGET_S = 1200.0


@dataclass(frozen=True)
class Verdict:
    rule: str
    level: str  # "pass", "fail", "warn", "note"
    detail: str

    @property
    def failed(self) -> bool:
        return self.level == "fail"


def admission_hash(target: TargetSpec) -> str:
    """What a check judged, without what later steps write back.

    Floors and docs come after admission, so writing them must not make a
    passing check look stale. Anything else changed, an input's setup most of
    all, and the check no longer describes the config.
    """
    d = target.to_dict()
    d.pop("docs")
    for i in d["inputs"]:
        i.pop("noise_floor")
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def call_max_s(repeats_per_launch: int) -> float:
    """The longest call ``repeats_per_launch`` timed calls leave room for."""
    return (LAUNCH_TIMEOUT - LAUNCH_HEADROOM_S) / repeats_per_launch


def timing_per_attempt_s(survey: Survey, pairs: int, repeats_per_launch: int) -> dict[str, float]:
    """Seconds of timing launches one attempt costs on each input that runs.

    Each pair launches the base and the patched tree once, and each launch makes
    ``repeats_per_launch`` calls after its fixed start cost.
    """
    return {
        i.name: pairs
        * 2
        * (
            repeats_per_launch * i.call_s
            + (i.fixed_s if i.fixed_s is not None else i.import_s + i.setup_s)
        )
        for i in survey.inputs
        if not i.error
    }


def judge(
    target: TargetSpec, survey: Survey, repeats_per_launch: int, pairs: int
) -> tuple[Verdict, ...]:
    out: list[Verdict] = []
    limit = call_max_s(repeats_per_launch)

    for i in survey.inputs:
        if i.error:
            out.append(Verdict(f"{i.name} runs", "fail", i.error))
            continue
        out.append(Verdict(f"{i.name} runs", "pass", f"call {i.call_s:.4f}s under cProfile"))
        if i.hot_executed:
            level = "warn" if i.hot_share < HOT_SHARE_WARN else "pass"
            note = (
                "" if level == "pass" else "; the agents will be optimising around something else"
            )
            out.append(
                Verdict(
                    f"{i.name} hot file executes",
                    level,
                    f"{target.hot_file} took {100 * i.hot_share:.1f}% of self time{note}",
                )
            )
        else:
            out.append(
                Verdict(
                    f"{i.name} hot file executes",
                    "fail",
                    f"{target.hot_file} did not run; the referee would refuse to time this input",
                )
            )
        if i.deterministic:
            out.append(
                Verdict(f"{i.name} deterministic", "pass", f"fingerprint {i.fingerprints[0]}")
            )
        else:
            out.append(
                Verdict(
                    f"{i.name} deterministic",
                    "fail",
                    f"two launches of the base tree gave {i.fingerprints[0]} and "
                    f"{i.fingerprints[1]}; result_matches cannot mean anything",
                )
            )
        if i.call_s < CALL_MIN_S:
            out.append(
                Verdict(
                    f"{i.name} call length",
                    "fail",
                    f"{i.call_s:.4f}s is under {CALL_MIN_S}s; a floor for it would be too loose to grade on",
                )
            )
        elif i.call_s > limit:
            out.append(
                Verdict(
                    f"{i.name} call length",
                    "fail",
                    f"{i.call_s:.1f}s exceeds {limit:.0f}s, the most {repeats_per_launch} repeats "
                    f"fit inside a {LAUNCH_TIMEOUT}s launch",
                )
            )
        else:
            out.append(Verdict(f"{i.name} call length", "pass", f"{i.call_s:.4f}s"))

    times = timing_per_attempt_s(survey, pairs, repeats_per_launch)
    if times:
        total = sum(times.values())
        slowest = ", ".join(
            f"{name} {s:.0f}s" for name, s in sorted(times.items(), key=lambda kv: -kv[1])[:3]
        )
        basis = (
            f"about {total:.0f}s of timing per attempt, {pairs} pairs of {repeats_per_launch} "
            f"calls per tree on each input, from call times under cProfile; slowest: {slowest}"
        )
        if total > REFEREE_TIMING_BUDGET_S:
            out.append(
                Verdict(
                    "referee time per attempt",
                    "fail",
                    f"{basis}. That exceeds {REFEREE_TIMING_BUDGET_S:.0f}s; make the slowest "
                    "inputs smaller or drop them",
                )
            )
        else:
            out.append(Verdict("referee time per attempt", "pass", basis))

    seen = {t.scope: t for t in survey.tests}
    for scope in ("module", "full"):
        t = seen.get(scope)
        if t is None:
            err = next((e for e in survey.errors if e.startswith(scope)), "no result recorded")
            out.append(Verdict(f"{scope} suite", "fail", err))
            continue
        if not t.ok:
            out.append(
                Verdict(
                    f"{scope} suite",
                    "fail",
                    f"{t.failed} failed, {t.errors} errors on the base tree; a patch could never pass",
                )
            )
        elif t.duration_s > TESTS_TIMEOUT - 60:
            out.append(
                Verdict(
                    f"{scope} suite",
                    "fail",
                    f"{t.duration_s:.0f}s exceeds the {TESTS_TIMEOUT - 60}s the referee allows",
                )
            )
        else:
            out.append(
                Verdict(f"{scope} suite", "pass", f"{t.passed} passed in {t.duration_s:.0f}s")
            )
    if target.tests.module == target.tests.full:
        out.append(
            Verdict(
                "two suites",
                "warn",
                "module and full name the same path, so the referee runs it twice per attempt",
            )
        )
    if target.uncalibrated:
        out.append(
            Verdict(
                "noise floors",
                "note",
                "uncalibrated: "
                + ", ".join(target.uncalibrated)
                + "; run autoresearch calibrate before a run",
            )
        )
    return tuple(out)


def render(target: TargetSpec, survey: Survey, verdicts: tuple[Verdict, ...]) -> str:
    lines = [f"target {target.name}: {target.package} at {target.sha[:12]}"]
    python = next((i.python for i in survey.inputs if i.python), "")
    if python:
        lines.append(f"python {python} in the box")
    lines.append("")
    lines.append(
        f"{'input':<16} {'call':>9} {'hot share':>9} {'import':>7} {'setup':>7} {'fixed':>7}  fingerprints"
    )
    for i in survey.inputs:
        if i.error:
            lines.append(f"{i.name:<16} error: {i.error}")
            continue
        fixed = "--" if i.fixed_s is None else f"{i.fixed_s:.2f}s"
        lines.append(
            f"{i.name:<16} {i.call_s:>8.4f}s {100 * i.hot_share:>8.1f}% {i.import_s:>6.2f}s "
            f"{i.setup_s:>6.2f}s {fixed:>7}  {i.fingerprints[0]} {i.fingerprints[1]}"
        )
    lines.append("")
    for t in survey.tests:
        lines.append(
            f"{t.scope} suite: {'ok' if t.ok else 'FAILED'}, {t.passed} passed, {t.failed} failed, "
            f"{t.errors} errors, {t.duration_s:.0f}s"
        )
    lines.append("")
    width = max(len(v.rule) for v in verdicts) if verdicts else 0
    for v in verdicts:
        lines.append(f"{v.level.upper():<5} {v.rule:<{width}}  {v.detail}")
    failed = [v for v in verdicts if v.failed]
    lines.append("")
    lines.append(
        "admissible" if not failed else f"not admissible: {len(failed)} rule(s) failed, named above"
    )
    return "\n".join(lines)
