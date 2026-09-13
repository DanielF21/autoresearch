"""The first user message and the per turn messages, rendered from the run.

The system prompt lives beside this module, one version per file (v1.py and on),
and is picked per worker slot. Everything here is the same for every version:
the target, the documents, the history index, the closing instruction, and the
two messages the loop injects (the nudge and the last turn).

Order of the first user message: the target, the documents about it, then the
history index. Each round only appends attempts, so the prefix of the next
round's prompt is the whole of this one. The per attempt instruction comes last.
"""

from __future__ import annotations

from autoresearch.config import BenchmarkInput, TargetSpec
from autoresearch.types import Attempt


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
    if m.untimed:
        # Named, so the reader knows the geomean beside it covers fewer inputs.
        return f"timing failed on {', '.join(m.untimed)}"
    if m.regressions:
        # Named, because which input a patch lost on is the whole lesson.
        return f"slower on {', '.join(m.regressions)}"
    if m.clears_noise:
        return "real speedup"
    return "below the noise floor"


SUMMARY_CHARS = 70
OUTCOME_CHARS = 28


def mechanism_line(rationale: str, limit: int = SUMMARY_CHARS) -> str:
    """The first non empty line of a rationale, cut to ``limit`` characters.

    The prompt asks every worker to make that line one sentence naming the
    mechanism, so the index becomes a map of what has been tried. A rationale
    from before that rule, or none at all, shows as what it is.
    """
    for line in rationale.splitlines():
        line = line.strip().lstrip("#-* ").strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
    return "(no rationale)"


def index_line(a: Attempt) -> str:
    m = a.measurement
    speed = "--" if m is None or m.speedup is None else f"{m.speedup:.2f}x"
    worst = "--" if m is None or m.worst_speedup is None else f"{m.worst_speedup:.2f}x"
    note = outcome(a)
    if a.duplicate_of:
        note += f", same diff as {a.duplicate_of}"
    return (
        f"{a.ref.dirname:<4}  {speed:>8}  {worst:>8}  {note:<{OUTCOME_CHARS}}  "
        f"{mechanism_line(a.rationale)}"
    )


def render_history(history: tuple[Attempt, ...]) -> str:
    """An index, not the attempts themselves.

    The attempts are on disk in the box, and the agent reads the ones it cares
    about with the shell. Rendering every diff here instead would put the whole
    history in every turn of every attempt, which at width 16 is hundreds of
    thousands of tokens per turn and buys nothing the filesystem does not. The
    last column is the one line every rationale opens with, so the whole table
    is a survey of mechanisms, not a leaderboard.
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
        "timing, which the two columns here only summarise. The last column is the "
        "first line of that attempt's rationale. Survey the whole table before you "
        "read any diff.\n\n"
        "```\n"
        f"{'id':<4}  {'geomean':>8}  {'worst':>8}  {'outcome':<{OUTCOME_CHARS}}  mechanism\n"
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
        f"You are attempt {attempt_number:04d}. Survey the history and the hot file, write "
        "down what you expect before you measure, then make your change."
    )


LAST_TURN = (
    "This is your last turn. Call submit now with the rationale of the diff currently in "
    "your working tree and the speedup you predicted for it. Any other tool call ends the "
    "attempt with the diff and no rationale."
)

NUDGE = (
    "You replied without calling a tool. Use the shell to look at the code and change it, "
    "run_tests and run_benchmark to check your work, and call submit when you have a patch. "
    "If you have nothing to submit, say so by calling submit with your best change so far."
)
