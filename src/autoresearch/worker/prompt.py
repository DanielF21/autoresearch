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

Your sandbox is a Debian container you are root in, with Python 3.12, git, and
the usual command line tools. The shell is how you look around and make changes.
There is no file reading or editing tool; use the shell for both. The filesystem:

  /workspace/repo      the repository, checked out at the original commit. This
                       is your working tree and the only thing you change. The
                       shell starts here.
  /workspace/base      the same commit, untouched, and read only. The benchmark
                       times your tree against it. Do not try to modify it.
  /workspace/history   every earlier attempt, one directory per attempt named
                       0001, 0002 and so on, each holding patch.diff, the full
                       measurement.json, and rationale.md in that worker's words.
                       The history table below is only an index of these.

How this works:
- Every attempt starts from the same original commit, and every attempt is
  measured against that same original. There is no accumulating version. If an
  earlier attempt found a gain you want to keep, include its diff in your own
  patch; every earlier patch is on disk under /workspace/history/NNNN/patch.diff
  and can be applied with git apply.
- The referee measures every submission in full and records all of it: the
  module's tests, the whole suite, whether the benchmark result is unchanged,
  six back to back timings against the original, and instruction counts. A
  submission that fails tests or is slower is still recorded, so the next
  worker learns from it.
- A real speedup is one that passes both suites, computes the same result, and
  whose measured speedup reaches the noise floor given below.
- Only source files may change. A patch that edits tests or benchmarks is
  measured but marked out of scope, and cannot count as a speedup.
- Every earlier attempt is in your history with its measurements. Read it
  before you start. Do not repeat a failed idea unless you can say what will be
  different.
- Work in small steps: look, change, run the module tests, run the benchmark,
  and submit when you have something. An attempt that never submits is wasted.
- When you submit, the harness takes git diff of your working tree as the
  patch. Nothing else you write counts.
"""


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def outcome(a: Attempt) -> str:
    """One phrase for what happened, most disqualifying fact first."""
    if a.skipped:
        # ``skipped`` is always "no_patch"; the stop reason is what says why.
        return f"no patch: {a.stop_reason}"
    m = a.measurement
    if m is None:
        return "not measured yet"
    if not m.applied:
        return "did not apply"
    if m.scope_violations:
        return "out of scope"
    if not m.tests_pass:
        return "tests failed"
    if m.result_matches is False:
        return "wrong result"
    if m.clears_noise:
        return "real speedup"
    return "below the noise floor"


def index_line(a: Attempt) -> str:
    m = a.measurement
    speed = "--" if m is None or m.speedup is None else f"{m.speedup:.2f}x"
    ir = "--" if m is None or m.ir is None else f"{m.ir.delta_pct:+.1f}%"
    note = outcome(a)
    if a.duplicate_of:
        note += f", same diff as {a.duplicate_of}"
    return f"{a.ref.dirname:<4}  {speed:>8}  {ir:>8}  {note}"


def render_history(history: tuple[Attempt, ...]) -> str:
    """An index, not the attempts themselves.

    The attempts are on disk in the box, and the agent reads the ones it cares
    about with the shell. Rendering every diff here instead would put the whole
    history in every turn of every attempt, which at width 16 is hundreds of
    thousands of tokens per turn and buys nothing the filesystem does not.
    """
    if not history:
        return "## History\n\nNo earlier attempts. You are first.\n"
    real = [a for a in history if a.clears_noise]
    best = max((a.measurement.speedup or 0.0 for a in real if a.measurement), default=None)
    head = (
        f"## History\n\n{_plural(len(history), 'earlier attempt')}, "
        + f"{_plural(len(real), 'real speedup')}"
        + (f", best {best:.2f}x" if best else "")
        + ". Newest last.\n\n"
        "Each is a directory under /workspace/history holding patch.diff, "
        "measurement.json and rationale.md. Read the ones worth reading before you "
        "repeat an idea or build on one.\n\n"
        "```\n"
        f"{'id':<4}  {'speedup':>8}  {'ir':>8}  outcome\n"
    )
    return head + "\n".join(index_line(a) for a in history) + "\n```\n"


def render_target(target: TargetSpec, base_sha: str, noise_floor: float) -> str:
    return (
        "## Target\n\n"
        f"Repository: {target.repo} at commit {base_sha[:12]}, the original for every attempt.\n"
        f"Hot file: {target.hot_file}\n"
        f"Its tests: {target.test_file}\n"
        f"Benchmark graph: `G = {target.graph}`\n"
        f"Benchmark call: `{target.call}`\n"
        "Speedup: the original's time divided by your patched time, taken as the median of "
        "six back to back pairs. 2.00x is twice as fast, 0.50x is half as fast, 1.00x is no "
        f"change. Below {noise_floor}x it cannot be told apart from a patch that changes "
        "nothing, so that is the floor a real speedup has to clear.\n"
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
        f"You are attempt {attempt_number:04d}. Begin by reading the hot file and the history "
        "with the shell, then make one change and submit."
    )


NUDGE = (
    "You replied without calling a tool. Use the shell to look at the code and change it, "
    "run_tests and run_benchmark to check your work, and call submit when you have a patch. "
    "If you have nothing to submit, say so by calling submit with your best change so far."
)
