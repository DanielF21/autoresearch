"""The documents a worker is shown about a target, made from a referee box's profile.

Three files, named after the target's primary input:

  profile_inputs.txt              one row per input: the call's wall time, the
                                  hot file's self time under cProfile and its
                                  share, and the input's setup; then the
                                  geometric mean of the shares, the same
                                  aggregate a patch is graded on.
  profile_<primary>_flat.txt      cProfile of the primary input, by self time.
  profile_<primary>_callers.txt   the callers view of the same profile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from autoresearch.config import TargetSpec
from autoresearch.referee.referee import InputProfile


def _head(target: TargetSpec) -> str:
    return f"at {target.package} {target.sha[:12]}, the base commit for every attempt."


def _one_line(setup: str) -> str:
    return " ".join(setup.split())


def inputs_document(target: TargetSpec, rows: Sequence[InputProfile]) -> str:
    lines = [
        f"Cost of {target.call} on each of the {len(rows)} inputs it is timed on,",
        _head(target),
        "",
        "Absolute times were taken on a referee box and will differ in yours. What",
        "carries over is the ratio between the inputs and where the time goes.",
        "'hot self' is the self time spent inside the hot file under cProfile,",
        "and 'hot share' is its share of all self time. A change with a fixed",
        "setup cost is free on the slowest input and can be the entire runtime",
        "of the fastest.",
        "",
        f"{'input':<16} {'call':>9} {'hot self':>9} {'hot share':>9}  setup",
    ]
    for r in rows:
        lines.append(
            f"{r.name:<16} {r.call_s:>8.4f}s {r.hot_s:>8.4f}s {100 * r.share:>8.1f}%  "
            f"{_one_line(r.setup)}"
        )
    shares = [r.share for r in rows if r.share > 0]
    geo = math.exp(sum(math.log(s) for s in shares) / len(shares)) if shares else 0.0
    lines += ["", f"geometric mean of the hot file's share: {100 * geo:.1f}%"]
    slowest = max(rows, key=lambda r: r.call_s)
    fastest = min(rows, key=lambda r: r.call_s)
    if fastest.call_s > 0 and slowest is not fastest:
        lines += [
            "",
            f"The call spans {slowest.call_s / fastest.call_s:.0f}x between "
            f"{slowest.name} and {fastest.name}.",
        ]
    lines += [
        "",
        "The flat and caller profiles in the other two documents are of the first",
        "input only.",
    ]
    return "\n".join(lines) + "\n"


def documents(target: TargetSpec, rows: Sequence[InputProfile]) -> dict[str, str]:
    """File name to text, in the order a worker is shown them."""
    if not rows:
        raise ValueError("no inputs were profiled")
    primary = rows[0]
    return {
        "profile_inputs.txt": inputs_document(target, rows),
        f"profile_{primary.name}_flat.txt": (
            f"cProfile of {target.call} with {_one_line(primary.setup)}\n"
            f"{_head(target)}\n" + primary.flat
        ),
        f"profile_{primary.name}_callers.txt": (
            f"Callers for the same profile: {target.call} with {_one_line(primary.setup)}\n"
            + primary.callers
        ),
    }


def write_documents(out_dir: Path, docs: dict[str, str]) -> tuple[Path, ...]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in docs.items():
        (out_dir / name).write_text(text)
        written.append(out_dir / name)
    return tuple(written)
