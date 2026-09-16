"""What a candidate and a meta prompt call are told.

A request is a system message and one user message. The user message opens
with a prefix that is identical for every slot, batch and call of a run, so the
endpoint's prefix cache holds it: the target rendered by the harness's own code
with the tool sentences off, the profile documents every harness worker gets,
and every file holding a block, whole, with the blocks marked. What varies
follows: for a candidate, the evolved instruction, the parent, and the programs
OpenEvolve's prompt shows; for a meta call, the instruction population.

The candidate's sections follow OpenEvolve's default templates
(openevolve/prompts/defaults/diff_user.txt, evolution_history.txt,
top_program.txt, inspirations_section.txt at 411fb59c): current program, previous
attempts, top programs, diverse programs, inspirations, the current program's
code, and the task with the SEARCH/REPLACE format and its example. The system
message opens with OpenEvolve's system_message.txt and adds the referee's rules,
taken verbatim from the harness's run rules so the two arms cannot drift on
what is measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from alphaevolve.blocks import MARK_END, MARK_START, Block, marked
from alphaevolve.database import FEATURE_DIMENSIONS
from alphaevolve.meta import MetaPrompt
from autoresearch.config import TargetSpec
from autoresearch.model.protocol import Message
from autoresearch.types import Attempt
from autoresearch.worker.prompt import common
from autoresearch.worker.prompt.render import outcome, render_docs, render_target

# Built rather than written, so no line of this file opens with a merge marker.
SEARCH = "<" * 7 + " SEARCH"
DIVIDER = "=" * 7
REPLACE = ">" * 7 + " REPLACE"

# The run rule bullets that describe the referee, not the harness worker's sandbox.
SHARED_BULLETS = ("The referee measures", "A real speedup", "The goal is", "Only source files")


def shared_rules() -> str:
    bullets = common.RUN_RULES.split("\n- ")[1:]
    picked = [b.rstrip() for b in bullets if b.startswith(SHARED_BULLETS)]
    if len(picked) != len(SHARED_BULLETS):
        raise RuntimeError("the harness run rules changed; the shared bullets are not all there")
    return "".join(f"- {b}\n" for b in picked)


def system_prompt() -> str:
    return (
        # OpenEvolve's system_message.txt.
        "You are an expert software developer tasked with iteratively improving a codebase.\n"
        "Your goal is to maximize the FITNESS SCORE while exploring diverse solutions across "
        "feature dimensions.\n"
        "The system maintains a collection of diverse programs; both high fitness AND "
        "diversity are valuable.\n\n"
        "The codebase is a Python library. A program is one version of the marked blocks in "
        "its source, and every program is measured against the original code by a referee:\n"
        + shared_rules()
        + "\nA program's fitness is its recorded speedup when it is a real speedup, and 0 "
        "otherwise. The original code has fitness 1.0000.\n"
    )


def constant_prefix(
    target: TargetSpec,
    base_sha: str,
    pairs: int,
    docs: tuple[tuple[str, str], ...],
    sources: dict[str, str],
    blocks: Sequence[Block],
) -> str:
    files = tuple(sorted({b.path for b in blocks}))
    parts = [
        render_target(target, base_sha, pairs, tools=False, editable=files),
        "\n",
        render_docs(docs),
        "\n## Source at the base commit\n\n",
        f"Each file below is shown whole. Only the code between {MARK_START} and "
        f"{MARK_END} may change; everything else is fixed. The programs further down are "
        "versions of these blocks.\n\n",
    ]
    for path in files:
        text = marked(sources[path], [b for b in blocks if b.path == path])
        parts.append(f"### {path}\n\n```python\n{text}```\n\n")
    return "".join(parts)


@dataclass(frozen=True)
class Shown:
    """A program as a prompt shows it. ``attempt`` is None for the base."""

    id: str
    fitness: float
    texts: tuple[str, ...]
    attempt: Attempt | None


def describe(p: Shown) -> str:
    a = p.attempt
    if a is None:
        return "The original code at the base commit, 1.00x by definition."
    lines = [f"Attempt {a.ref.dirname}: {outcome(a)}."]
    m = a.measurement
    if m is not None:
        if m.apply_error:
            lines.append(f"Apply error: {m.apply_error[:300]}")
        for i in m.inputs:
            held = "untimed" if i.speedup is None else f"{i.speedup:.3f}x"
            shown = "" if i.shown_speedup is None else f", shown seed {i.shown_speedup:.3f}x"
            lines.append(f"- {i.name}: held out seed {held}{shown}")
        failed = [t for suite in m.tests for t in suite.failed_tests]
        if failed:
            lines.append("Failed tests: " + ", ".join(failed[:5]))
    return "\n".join(lines)


def _code(p: Shown, blocks: Sequence[Block]) -> str:
    return "".join(
        f"\n#### {b.path}, lines {b.start} to {b.end} at the base commit: {b.function}\n"
        f"```python\n{text}```\n"
        for b, text in zip(blocks, p.texts, strict=True)
    )


def _program(title: str, p: Shown, blocks: Sequence[Block]) -> str:
    return f"### {title} (Score: {p.fitness:.4f})\n{describe(p)}\n{_code(p, blocks)}\n"


TASK = f"""# Task
Suggest improvements to the program that will improve its FITNESS SCORE.
The system maintains diversity across these dimensions: {", ".join(FEATURE_DIMENSIONS)}
Different solutions with similar fitness but different features are valuable.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

{SEARCH}
# Original code to find and replace (must match exactly)
{DIVIDER}
# New replacement code
{REPLACE}

Example of valid diff format:
{SEARCH}
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
{DIVIDER}
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
{REPLACE}

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Here every SEARCH must match exactly one place across the current program's blocks, and never the marker lines.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program; focus on targeted improvements.
"""


def variable_part(
    meta_text: str,
    parent: Shown,
    previous: Sequence[Shown],
    top: Sequence[Shown],
    diverse: Sequence[Shown],
    inspirations: Sequence[Shown],
    blocks: Sequence[Block],
) -> str:
    out = [
        "## Instructions evolved in this run\n\n",
        (meta_text or "(none yet)") + "\n\n",
        "# Current Program Information\n",
        f"- Fitness: {parent.fitness:.4f}\n",
        f"- Program: {parent.id}\n",
        describe(parent) + "\n\n",
        "# Program Evolution History\n## Previous Attempts\n\n",
    ]
    for i, p in enumerate(previous):
        first = describe(p).splitlines()[0]
        out.append(
            f"### Attempt {len(previous) - i}\n- Program: {p.id}\n- Fitness: {p.fitness:.4f}\n"
            f"- Outcome: {first}\n\n"
        )
    out.append("## Top Performing Programs\n\n")
    out.extend(_program(f"Program {i + 1}", p, blocks) for i, p in enumerate(top))
    if diverse:
        out.append("## Diverse Programs\n\n")
        out.extend(_program(f"Program D{i + 1}", p, blocks) for i, p in enumerate(diverse))
    if inspirations:
        out.append(
            "## Inspiration Programs\n\nThese programs represent diverse approaches and "
            "creative solutions that may inspire new ideas:\n\n"
        )
        out.extend(_program(f"Inspiration {i + 1}", p, blocks) for i, p in enumerate(inspirations))
    out.append("# Current Program\n" + _code(parent, blocks) + "\n")
    out.append(TASK)
    return "".join(out)


def candidate_messages(system: str, prefix: str, variable: str) -> list[Message]:
    return [{"role": "system", "content": system}, {"role": "user", "content": prefix + variable}]


def meta_part(pool: Sequence[MetaPrompt]) -> str:
    out = [
        "# Task: write instructions\n\n",
        "Each engineer proposing a change to the blocks above receives one set of "
        "instructions, drawn from those below in proportion to how well the programs written "
        "under it scored. Write one new set that you expect to lead to faster programs: what "
        "to look for in the profile and the source, which kinds of change pay here, and which "
        "to avoid. Build on what scored well, and do not repeat an existing set.\n\n",
    ]
    for p in pool:
        best = "none yet" if p.best is None else f"{p.best:.4f}"
        out.append(
            f"### {p.id}: used by {p.uses} candidates, best fitness {best}\n"
            f"{p.text or '(empty: no instructions beyond the task)'}\n\n"
        )
    out.append(
        "Reply with the new instructions between <instructions> and </instructions>, "
        "in under 200 words.\n"
    )
    return "".join(out)


def meta_messages(system: str, prefix: str, pool: Sequence[MetaPrompt]) -> list[Message]:
    return candidate_messages(system, prefix, meta_part(pool))
