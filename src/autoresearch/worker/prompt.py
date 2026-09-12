"""What the worker is told. Stable content first, so the prefix caches.

Order of the first user message: the target, the documents about it, then every
earlier attempt in full. Each round only appends attempts, so the prefix of the
next round's prompt is the whole of this one. The per attempt instruction comes
last.
"""

from __future__ import annotations

from autoresearch.config import TargetSpec
from autoresearch.types import Attempt

SYSTEM_PROMPT = """\
You are a performance engineer working alone in a sandbox on one repository.
Your job is to make one benchmark faster with one patch, without changing what
the code computes, and to say before measurement how much faster you expect it
to be.

How this works:
- Every attempt starts from the same original commit, and every attempt is
  measured against that same original. There is no accumulating version. If an
  earlier attempt found a gain you want to keep, include its diff in your own
  patch; every earlier patch is on disk under /workspace/history/NNNN/patch.diff
  and can be applied with the shell tool.
- The referee measures every submission in full and records all of it: the
  module's tests, the whole suite, whether the benchmark result is unchanged,
  six back to back timings against the original, and instruction counts. A
  submission that fails tests or is slower is still recorded, so the next
  worker learns from it.
- A real speedup is one that passes both suites, computes the same result, and
  whose median timing ratio reaches the noise floor. Below the floor, the
  timing cannot be told apart from a patch that changes nothing.
- Only source files may change. A patch that edits tests or benchmarks is
  measured but marked out of scope, and cannot count as a speedup.
- Every earlier attempt is in your history with its measurements. Read it
  before you start. Do not repeat a failed idea unless you can say what will be
  different.
- Work in small steps: read, change, run the module tests, run the benchmark,
  and submit when you have something. An attempt that never submits is wasted.
- When you submit, the harness takes git diff of your working tree as the
  patch. Nothing else you write counts.
"""


def _fmt_ratio(value: float | None) -> str:
    return "not timed" if value is None else f"{value:.4f}"


def render_attempt(a: Attempt) -> str:
    lines = [f"### Attempt {a.ref.dirname} (round {a.ref.round}, worker {a.ref.worker})"]
    if a.duplicate_of:
        lines.append(f"same diff as attempt {a.duplicate_of}")
    pred = "none" if a.prediction is None else f"{a.prediction.speedup:.3f}"
    m = a.measurement
    if a.skipped:
        lines.append(f"not measured: {a.skipped}")
    elif m is None:
        lines.append("measurement: pending")
    else:
        tests = (
            ", ".join(
                f"{t.scope} {'pass' if t.ok else 'FAIL'} ({t.passed} passed, {t.failed} failed)"
                for t in m.tests
            )
            or "not run"
        )
        match = {True: "same", False: "DIFFERENT", None: "unknown"}[m.result_matches]
        lines += [
            f"applied: {'yes' if m.applied else 'no: ' + m.apply_error}",
            f"tests: {tests}",
            f"benchmark result: {match}",
            f"median ratio vs original: {_fmt_ratio(m.median_ratio)}; predicted: {pred}",
            f"real speedup (clears noise floor {m.noise_floor}): {'yes' if m.clears_noise else 'no'}",
        ]
        if m.ir is not None:
            lines.append(f"instruction count change: {m.ir.delta_pct:+.2f} percent")
        if m.scope_violations:
            lines.append("out of scope: " + "; ".join(m.scope_violations))
        if m.errors:
            lines.append("measurement errors: " + "; ".join(m.errors))
    lines.append(f"stop reason: {a.stop_reason}")
    if a.rationale.strip():
        lines += ["rationale:", a.rationale.strip()]
    if a.patch:
        lines += [
            f"patch (also at /workspace/history/{a.ref.dirname}/patch.diff):",
            "```diff",
            a.patch.rstrip(),
            "```",
        ]
    else:
        lines.append("patch: none")
    return "\n".join(lines)


def render_history(history: tuple[Attempt, ...]) -> str:
    if not history:
        return "## History\n\nNo earlier attempts. You are first.\n"
    real = [a for a in history if a.clears_noise]
    best = max((a.measurement.median_ratio or 0.0 for a in real if a.measurement), default=None)
    head = (
        f"## History\n\n{len(history)} earlier attempts, {len(real)} real speedups"
        + (f", best median ratio {best:.4f}" if best else "")
        + ". Newest last.\n"
    )
    return head + "\n\n".join(render_attempt(a) for a in history) + "\n"


def render_target(target: TargetSpec, base_sha: str, noise_floor: float) -> str:
    return (
        "## Target\n\n"
        f"Repository: {target.repo} at commit {base_sha[:12]}, the original for every attempt.\n"
        f"Hot file: {target.hot_file}\n"
        f"Its tests: {target.test_file}\n"
        f"Benchmark graph: `G = {target.graph}`\n"
        f"Benchmark call: `{target.call}`\n"
        f"Noise floor: a median ratio of original time to patched time below {noise_floor} "
        "cannot be told apart from no change.\n"
        f"Allowed files: {', '.join(target.allow)}. Never: {', '.join(target.deny)}.\n"
    )


def render_docs(docs: tuple[tuple[str, str], ...]) -> str:
    if not docs:
        return ""
    parts = ["## Documents\n"]
    for name, text in docs:
        parts.append(f"### {name}\n\n```\n{text.rstrip()}\n```\n")
    return "\n".join(parts)


def initial_user_message(
    target: TargetSpec,
    base_sha: str,
    noise_floor: float,
    docs: tuple[tuple[str, str], ...],
    history: tuple[Attempt, ...],
    attempt_number: int,
) -> str:
    return (
        render_target(target, base_sha, noise_floor)
        + "\n"
        + render_docs(docs)
        + "\n"
        + render_history(history)
        + "\n## This attempt\n\n"
        f"You are attempt {attempt_number:04d}. The repository is checked out at the original "
        f"commit in your sandbox. Earlier attempts are also on disk under /workspace/history/NNNN/. "
        "Begin by reading the hot file and the history, then make one change and submit."
    )


NUDGE = (
    "You replied without calling a tool. Use the tools to read, edit, test and benchmark, "
    "and call submit when you have a patch. If you have nothing to submit, say so by "
    "calling submit with your best change so far."
)
