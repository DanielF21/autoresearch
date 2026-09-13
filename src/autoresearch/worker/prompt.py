"""What the worker is told. Stable content first, so the prefix caches.

Order of the first user message: the target, the documents about it, then every
earlier attempt in full. Each round only appends attempts, so the prefix of the
next round's prompt is the whole of this one. The per attempt instruction comes
last.
"""

from __future__ import annotations

from autoresearch.config import BenchmarkInput, TargetSpec
from autoresearch.types import Attempt

_SYSTEM_PROMPT = """\
You are a performance engineer working alone in a sandbox on one repository.
Your job is to make one call faster with one patch, across every input it is
measured on, without changing what the code computes, and to say before
measurement how much faster you expect it to be.

Your sandbox is a Debian container you are root in, with Python {python}, git,
and the usual command line tools. The shell is how you look around and make changes.
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
  module's tests, the whole suite, whether the result is unchanged on every
  input, and back to back timing pairs on every input against the original. How
  many pairs, and on which inputs, is stated with the target below. A submission
  that fails tests or is slower is still recorded, so the next worker learns
  from it.
- A real speedup passes both suites, computes the same result on every input,
  is not slower on any input, and clears the noise floor on at least one. The
  number recorded for it is the geometric mean across the inputs.
- Only source files may change. A patch that edits tests or benchmarks is
  measured but marked out of scope, and cannot count as a speedup.
- Every earlier attempt is in your history with its measurements. Read it
  before you start. Do not repeat a failed idea unless you can say what will be
  different.
- Work in small steps: look, change, run the module tests, run the benchmark,
  and submit when you have something. An attempt that never submits is wasted.
- run_benchmark times every input, so it is what tells you whether a change
  helps everywhere or only where you were looking. Run it before you submit.
- run_tests runs the hot module's own tests, which is the check worth making
  while you work. You cannot run the whole suite and do not need to: the referee
  runs it on every submission, and a submission that breaks it is recorded as
  failing tests. Spend the time on the patch instead.
- When you submit, the harness takes git diff of your working tree as the
  patch. Nothing else you write counts.
"""


def system_prompt(python: str) -> str:
    """The system message. ``python`` is the version the box reported, never assumed."""
    return _SYSTEM_PROMPT.format(python=python)


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
    if m.regressions:
        # Named, because which input a patch lost on is the whole lesson.
        return f"slower on {', '.join(m.regressions)}"
    if m.clears_noise:
        return "real speedup"
    return "below the noise floor"


def index_line(a: Attempt) -> str:
    m = a.measurement
    speed = "--" if m is None or m.speedup is None else f"{m.speedup:.2f}x"
    worst = "--" if m is None or m.worst_speedup is None else f"{m.worst_speedup:.2f}x"
    note = outcome(a)
    if a.duplicate_of:
        note += f", same diff as {a.duplicate_of}"
    return f"{a.ref.dirname:<4}  {speed:>8}  {worst:>8}  {note}"


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
        "measurement.json and rationale.md. The measurement holds every input's own "
        "timing, which the two columns here only summarise. Read the ones worth reading "
        "before you repeat an idea or build on one.\n\n"
        "```\n"
        f"{'id':<4}  {'geomean':>8}  {'worst':>8}  outcome\n"
    )
    return head + "\n".join(index_line(a) for a in history) + "\n```\n"


def _scoring_rule(n: int) -> str:
    """How the n input ratios become the one number, in that target's arithmetic.

    The worked example is computed rather than written out, because the input
    count is a property of the config. A target with one input has no mean to
    take and gets a sentence that says so.
    """
    if n == 1:
        return (
            "**Your recorded speedup is the ratio on that one input.** There is no mean to "
            "take.\n\n"
        )
    spike = 100.0 ** (1.0 / n)
    doubled = 2.0 ** (1.0 / n)
    return (
        f"**Your recorded speedup is the geometric mean across all {n}.** A patch that is "
        f"100x on one input and unchanged on the other {n - 1} scores {spike:.2f}x, the same "
        f"as one that is {spike:.2f}x everywhere. Doubling every input doubles your score; "
        f"doubling one of {n} multiplies it by {doubled:.3f}. Breadth counts for {n} times "
        "what depth counts for.\n\n"
    )


def _input_block(i: BenchmarkInput) -> str:
    floor = "uncalibrated" if i.noise_floor is None else f"floor {i.noise_floor:.4f}"
    setup = "\n".join(f"    {line}" for line in i.setup.strip().splitlines())
    return f"{i.name}  ({floor})\n{setup}"


def render_target(target: TargetSpec, base_sha: str, pairs: int) -> str:
    package_at = (
        f"{target.package_root}/{target.package}" if target.package_root != "." else target.package
    )
    return (
        "## Target\n\n"
        f"Repository: {target.repo} at commit {base_sha[:12]}, the original for every attempt.\n"
        f"Package: `{target.package}`, imported from `{package_at}` in the tree, bound to "
        f"`{target.alias}` below.\n"
        f"Hot file: {target.hot_file}\n"
        f"Its tests: {target.tests.module}\n"
        f"Benchmark call: `{target.call}`\n"
        f"Allowed files: {', '.join(target.allow)}. Never: {', '.join(target.deny)}.\n"
        "\n### Inputs\n\n"
        f"The same call is timed on every one of these, {_plural(pairs, 'back to back pair')} "
        "each, on every submission. They are not variations to pick between: your patch is "
        "measured on all of them. Each input is the statements below, run once with "
        f"`{target.alias}` bound to the package and `ROOT` to the tree, and then the call "
        "is timed in that namespace.\n\n"
        "```\n" + "\n\n".join(_input_block(i) for i in target.inputs) + "\n```\n\n"
        "Speedup on one input is the original's time divided by your patched time, taken as "
        "the median over that input's pairs. 2.00x is twice as fast, 0.50x is half as fast, "
        "1.00x is no change. Each input has its own floor above, because a call of a few "
        "milliseconds is noisier than one of a second; below its floor a change cannot be "
        "told apart from a patch that does nothing.\n\n"
        + _scoring_rule(len(target.inputs))
        + "**A patch that is slower on any input is not a speedup at all**, however fast it "
        "is elsewhere. Slower means below 1/floor for that input, the mirror of the test "
        "above. This is the rule a fast path tends to fall foul of: setup cost is paid on "
        "every call, so a path that pays for itself where the call is expensive can lose "
        "where the call is cheap, and a guard on the wrong property will not prevent that. "
        "run_benchmark times every input, so run it before you submit.\n"
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
    docs: tuple[tuple[str, str], ...],
    history: tuple[Attempt, ...],
    attempt_number: int,
    pairs: int,
) -> str:
    return (
        render_target(target, base_sha, pairs)
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
