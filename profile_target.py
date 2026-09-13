#!/usr/bin/env python3
"""The documents a worker is shown about a target, generated from its config.

    uv run profile_target.py configs/t1_w4d.toml --repo /path/to/checkout --out configs/docs

Three files, named after the target's primary input:

  profile_inputs.txt              one row per input: the call's wall time, the
                                  hot file's self time under cProfile and its
                                  share, and the input's setup; then the
                                  geometric mean of the shares, the same
                                  aggregate a patch is graded on.
  profile_<primary>_flat.txt      cProfile of the primary input, by self time.
  profile_<primary>_callers.txt   the callers view of the same profile.

Local and box free: the checkout at ``--repo`` must be at the config's sha and
the target's dependencies must be importable by this interpreter, since the
package is imported by path from that checkout exactly as the guest does it.
Absolute times are this machine's and the documents say so; what a worker
reads off them is the ratio between inputs and where the time goes.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import importlib
import io
import math
import pathlib
import pstats
import subprocess
import sys
import time
from typing import Any

from autoresearch.config import BenchmarkInput, TargetSpec, load_config

FLAT_ROWS = 14
CALLER_ROWS = 10
PLAIN_RUNS = 3


def _import_by_path(target: TargetSpec, repo: pathlib.Path) -> Any:
    sys.path.insert(0, str((repo / target.package_root).resolve()))
    module = importlib.import_module(target.package)
    where = pathlib.Path(str(module.__file__)).resolve()
    if repo.resolve() not in where.parents:
        raise SystemExit(f"{target.package} resolved to {where}, outside {repo}")
    return module


def _namespace(
    target: TargetSpec, module: Any, repo: pathlib.Path, inp: BenchmarkInput
) -> dict[str, Any]:
    scope: dict[str, Any] = {target.alias: module, "ROOT": repo.resolve()}
    exec(inp.setup, scope)
    return scope


def _plain_min_s(call: str, scope: dict[str, Any]) -> float:
    times = []
    for _ in range(PLAIN_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        eval(call, scope)
        times.append(time.perf_counter() - t0)
    return min(times)


def _profile(call: str, scope: dict[str, Any]) -> pstats.Stats:
    gc.collect()
    profiler = cProfile.Profile()
    profiler.enable()
    eval(call, scope)
    profiler.disable()
    return pstats.Stats(profiler)


def _hot_self_s(stats: pstats.Stats, hot: pathlib.Path) -> tuple[float, float]:
    """Self seconds inside the hot file, and the profile's total self seconds."""
    hot_tt = 0.0
    total = 0.0
    for (fname, _line, _fn), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        total += tt
        if fname and not fname.startswith(("<", "~")):
            try:
                if pathlib.Path(fname).resolve() == hot:
                    hot_tt += tt
            except OSError:
                continue
    return hot_tt, total


def _relative(text: str, repo: pathlib.Path) -> str:
    return text.replace(str(repo.resolve()) + "/", "")


def _render(stats: pstats.Stats, view: str, rows: int, repo: pathlib.Path) -> str:
    out = io.StringIO()
    stats.stream = out  # type: ignore[attr-defined]
    stats.sort_stats("tottime")
    if view == "flat":
        stats.print_stats(rows)
    else:
        stats.print_callers(rows)
    return _relative(out.getvalue(), repo)


def _head(target: TargetSpec, sha: str) -> str:
    return f"at {target.package} {sha[:12]}, the base commit for every attempt."


def inputs_document(target: TargetSpec, rows: list[dict[str, Any]], sha: str) -> str:
    n = len(rows)
    lines = [
        f"Cost of {target.call} on each of the {n} inputs it is timed on,",
        _head(target, sha),
        "",
        "Absolute times are from one machine and will differ in your box. What",
        "carries over is the ratio between the inputs and where the time goes.",
        "'hot self' is the self time spent inside the hot file under cProfile,",
        "and 'hot share' is its share of all self time. A change with a fixed",
        "setup cost is free on the slowest input and can be the entire runtime",
        "of the fastest.",
        "",
        f"{'input':<16} {'call':>9} {'hot self':>9} {'hot share':>9}  setup",
    ]
    for r in rows:
        setup = " ".join(str(r["setup"]).split())
        lines.append(
            f"{r['name']:<16} {r['call_s']:>8.4f}s {r['hot_s']:>8.4f}s {100 * r['share']:>8.1f}%  {setup}"
        )
    shares = [r["share"] for r in rows if r["share"] > 0]
    geo = math.exp(sum(math.log(s) for s in shares) / len(shares)) if shares else 0.0
    lines += ["", f"geometric mean of the hot file's share: {100 * geo:.1f}%"]
    slowest = max(rows, key=lambda r: r["call_s"])
    fastest = min(rows, key=lambda r: r["call_s"])
    if fastest["call_s"] > 0 and slowest is not fastest:
        lines += [
            "",
            f"The call spans {slowest['call_s'] / fastest['call_s']:.0f}x between "
            f"{slowest['name']} and {fastest['name']}.",
        ]
    lines += [
        "",
        "The flat and caller profiles in the other two documents are of the first",
        "input only.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--repo", required=True, help="a checkout of the target at the config's sha")
    p.add_argument("--out", default="configs/docs")
    args = p.parse_args(argv)

    cfg = load_config(pathlib.Path(args.config))
    target = cfg.target
    repo = pathlib.Path(args.repo).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != target.sha:
        raise SystemExit(f"{repo} is at {head}, not the config's {target.sha}")
    module = _import_by_path(target, repo)
    hot = (repo / target.hot_file).resolve()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    primary_stats: pstats.Stats | None = None
    for inp in target.inputs:
        scope = _namespace(target, module, repo, inp)
        call_s = _plain_min_s(target.call, scope)
        stats = _profile(target.call, scope)
        hot_s, total = _hot_self_s(stats, hot)
        rows.append(
            {
                "name": inp.name,
                "setup": inp.setup,
                "call_s": call_s,
                "hot_s": hot_s,
                "share": hot_s / total if total else 0.0,
            }
        )
        if primary_stats is None:
            primary_stats = stats
        print(f"{inp.name:<16} call {call_s:.4f}s  hot share {100 * rows[-1]['share']:.1f}%")

    assert primary_stats is not None
    primary = target.primary
    (out / "profile_inputs.txt").write_text(inputs_document(target, rows, target.sha))
    flat = (
        f"cProfile of {target.call} with {' '.join(primary.setup.split())}\n"
        f"{_head(target, target.sha)}\n" + _render(primary_stats, "flat", FLAT_ROWS, repo)
    )
    (out / f"profile_{primary.name}_flat.txt").write_text(flat)
    callers = (
        f"Callers for the same profile: {target.call} with {' '.join(primary.setup.split())}\n"
        + _render(primary_stats, "callers", CALLER_ROWS, repo)
    )
    (out / f"profile_{primary.name}_callers.txt").write_text(callers)
    for name in (
        "profile_inputs.txt",
        f"profile_{primary.name}_flat.txt",
        f"profile_{primary.name}_callers.txt",
    ):
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
