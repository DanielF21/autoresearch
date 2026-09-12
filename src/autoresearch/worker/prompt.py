"""What the worker is told. Stable content first, so the prefix caches.

Order of the first user message: the target, the documents about it, then every
earlier attempt in full. Each round only appends attempts, so the prefix of the
next round's prompt is the whole of this one. The per attempt instruction comes
last.
"""

from __future__ import annotations

from autoresearch.config import TargetSpec
from autoresearch.types import Attempt, Verdict

SYSTEM_PROMPT = """\
You are a performance engineer working alone in a sandbox on one repository.
Your job is to make one benchmark faster with one patch, without changing what
the code computes, and to say before measurement how much faster you expect it
to be.

Rules:
- Correctness first. The referee runs the module's tests and then the whole
  suite. A patch that fails either is rejected without being timed.
- Only source files may change. Editing tests or benchmarks is rejected as
  tampering.
- The referee times your tree against the current incumbent, six back to back
  pairs, and accepts only if the median speedup clears the threshold. Small
  real speedups pass; noise does not.
- Every earlier attempt is in your history, accepted and rejected alike, with
  the referee's numbers. Read it before you start. Do not repeat a rejected
  idea unless you can say what will be different.
- Work in small steps: read, change, run the module tests, run the benchmark,
  and submit when you have something. An attempt that never submits is wasted.
- When you submit, the harness takes git diff of your working tree as the
  patch. Nothing else you write counts.
"""


def _ratio(a: Attempt) -> str:
    if a.result is None or a.result.median_ratio is None:
        return "not timed"
    return f"{a.result.median_ratio:.4f}"


def render_attempt(a: Attempt) -> str:
    verdict = "pending" if a.verdict is None else str(a.verdict)
    reason = "" if a.result is None else a.result.reason
    pred = "none" if a.prediction is None else f"{a.prediction.speedup:.3f}"
    lines = [
        f"### Attempt {a.ref.dirname} (round {a.ref.round}, worker {a.ref.worker})",
        f"verdict: {verdict}",
        f"referee median ratio: {_ratio(a)}; predicted: {pred}",
    ]
    if reason:
        lines.append(f"reason: {reason}")
    if a.result is not None and a.result.ir is not None:
        lines.append(f"instruction count change: {a.result.ir.delta_pct:+.2f} percent")
    lines.append(f"stop reason: {a.stop_reason}")
    if a.rationale.strip():
        lines += ["rationale:", a.rationale.strip()]
    if a.patch:
        lines += ["patch:", "```diff", a.patch.rstrip(), "```"]
    else:
        lines.append("patch: none")
    return "\n".join(lines)


def render_history(history: tuple[Attempt, ...]) -> str:
    if not history:
        return "## History\n\nNo earlier attempts. You are first.\n"
    accepted = sum(1 for a in history if a.verdict == Verdict.ACCEPTED)
    head = f"## History\n\n{len(history)} earlier attempts, {accepted} accepted. Newest last.\n"
    return head + "\n\n".join(render_attempt(a) for a in history) + "\n"


def render_target(target: TargetSpec, incumbent_sha: str, threshold: float) -> str:
    return (
        "## Target\n\n"
        f"Repository: {target.repo} at commit {incumbent_sha[:12]} plus any accepted patches.\n"
        f"Hot file: {target.hot_file}\n"
        f"Its tests: {target.test_file}\n"
        f"Benchmark graph: `G = {target.graph}`\n"
        f"Benchmark call: `{target.call}`\n"
        f"Acceptance threshold: median ratio of incumbent time to patched time at least {threshold}.\n"
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
    incumbent_sha: str,
    threshold: float,
    docs: tuple[tuple[str, str], ...],
    history: tuple[Attempt, ...],
    attempt_number: int,
) -> str:
    return (
        render_target(target, incumbent_sha, threshold)
        + "\n"
        + render_docs(docs)
        + "\n"
        + render_history(history)
        + "\n## This attempt\n\n"
        f"You are attempt {attempt_number:04d}. The repository is checked out at the incumbent "
        f"in your sandbox. Earlier attempts are also on disk under /workspace/history/NNNN/. "
        "Begin by reading the hot file and the history, then make one change and submit."
    )


NUDGE = (
    "You replied without calling a tool. Use the tools to read, edit, test and benchmark, "
    "and call submit when you have a patch. If you have nothing to submit, say so by "
    "calling submit with your best change so far."
)
