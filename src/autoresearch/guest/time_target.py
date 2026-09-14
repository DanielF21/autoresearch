"""One launch of the target against one source tree. Prints one JSON object.

The tree is imported by path, never installed, so two trees can be timed in
turn from the same box. The launch refuses to run if the package resolves
outside the requested tree.

Everything about the target comes from the flags; nothing is defaulted here.
``--setup`` is Python statements run once in a namespace holding the package
under ``--alias``, ``ROOT``, a ``pathlib.Path`` of the tree, and ``SEED``, the
integer from ``--seed`` that picks which instance of the input the setup
builds. ``--call`` is one expression evaluated in that namespace, and that is
what is timed.

The timed region is the call and a walk over its result: every mapping item,
set member, sequence element and array element is touched before the clock
stops. A lazy container that does its work when first read is then paid where
it is measured, the way a generator has to be wrapped in ``list()``. The walk
is the fingerprint's own traversal without the rendering, and its cost is
reported apart as ``walk_s``.

The result is fingerprinted so both trees can be shown to compute the same
thing. The fingerprint is a hash of a canonical rendering: mappings in key
order, floats rounded, sets sorted, array like objects through ``tolist``. An
object whose only rendering is an address cannot be compared across launches
and is an error, as is a result that is an iterator, since timing a call that
returns a generator times nothing. The whole result is hashed, never a slice
of it: pyparsing's seeded config once fingerprinted the first 20 and last 5
tokens of a 10 000 token parse, and a patch that got the middle wrong would
have passed as the same result.

Modes:

- run (default): ``--repeats`` timed calls, each wrapped in the provenance
  counters so a sample that saw steal, throttling, a page fault, a CPU
  migration or memory pressure is marked contaminated.
- ``--verify``: profile one call and report whether the hot file executed and
  what share of self time it took. Also warms the ``.pyc`` so no later launch
  pays compilation.
- ``--profile``: the plain minimum of three calls, then one call under cProfile,
  with the hot file's self time and the flat and callers views as text. What
  the profile documents a worker reads are made from.

Both modes report ``fixed_s``, the age of this process when the first timed
call began: interpreter start, the import and the setup. It is read from
``/proc`` and is None where there is none.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence, Set
from typing import Any

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

FLOAT_PLACES = 9
PROFILE_PLAIN_RUNS = 3


class UnfingerprintableError(ValueError):
    """A result with no rendering that two launches could agree on."""


def canonical(value: Any) -> str:
    """A rendering of ``value`` that depends on its content and nothing else."""
    if value is None or isinstance(value, bool | int | str | bytes):
        return repr(value)
    if isinstance(value, float):
        return repr(round(value, FLOAT_PLACES))
    if isinstance(value, Mapping):
        items = sorted((canonical(k), canonical(v)) for k, v in value.items())
        mapping = "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
        # A value that is both a mapping and a sequence renders as both: pyparsing's
        # ParseResults registers as each, and its named results are usually empty
        # while its tokens are the whole answer. Rendering only the mapping hashed
        # every parse as "{}" and no seed changed it.
        if isinstance(value, Sequence):
            return mapping + "[" + ",".join(canonical(v) for v in value) + "]"
        return mapping
    if isinstance(value, Set):
        return "{" + ",".join(sorted(canonical(v) for v in value)) + "}"
    if isinstance(value, Sequence):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    if hasattr(value, "__next__"):
        raise UnfingerprintableError(
            "the result is an iterator, so the timed call did no work; "
            "wrap the call in list() or otherwise consume it"
        )
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return canonical(to_list())
    text = repr(value)
    if " at 0x" in text:
        raise UnfingerprintableError(
            f"the result renders as an address ({text[:60]}), which no two launches "
            "share; have the call return plain data such as lists, dicts, strings and numbers"
        )
    return text


def fingerprint(value: Any) -> str:
    """A short hash of the canonical rendering. Raises UnfingerprintableError."""
    return hashlib.md5(canonical(value).encode()).hexdigest()[:10]


def materialize(value: Any) -> None:
    """Touch every element ``canonical`` would render, building nothing.

    Run inside the timed region, so a result that defers its work until it is
    read pays for that work where it is measured. The traversal is the
    fingerprint's, so what is forced here is exactly what is later compared. An
    iterator is refused here for the reason it is refused there.
    """
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            materialize(k)
            materialize(v)
        if not isinstance(value, Sequence):
            return
    if isinstance(value, Set | Sequence):
        for v in value:
            materialize(v)
        return
    if hasattr(value, "__next__"):
        raise UnfingerprintableError(
            "the result is an iterator, so the timed call did no work; "
            "wrap the call in list() or otherwise consume it"
        )
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        materialize(to_list())
        return
    # Anything else the fingerprint renders with repr, which reads it whole.
    repr(value)


def process_age_s() -> float | None:
    """Seconds since this interpreter started, from /proc. None without /proc."""
    try:
        stat = pathlib.Path("/proc/self/stat").read_text()
        uptime = float(pathlib.Path("/proc/uptime").read_text().split()[0])
    except OSError:
        return None
    # The command name (field 2) may contain spaces; everything after its
    # closing parenthesis is fixed width. Field 22, starttime, is then index 19.
    fields = stat.rsplit(")", 1)[1].split()
    start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
    return uptime - start


def _load_provenance() -> Any:
    """The sibling provenance module, found through this file's own directory."""
    import provenance  # type: ignore[import-not-found]

    return provenance


def _pyc_fresh(hot: pathlib.Path) -> bool:
    pyc = pathlib.Path(importlib.util.cache_from_source(str(hot)))
    return pyc.exists() and pyc.stat().st_mtime >= hot.stat().st_mtime


def hot_self_time(stats: Any, hot: pathlib.Path) -> tuple[float, float, bool]:
    """Self seconds inside ``hot``, the profile's total self seconds, and whether it ran."""
    executed = False
    hot_tt = 0.0
    total = 0.0
    for (fname, _line, _fn), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        total += tt
        if fname and not fname.startswith(("<", "~")):
            try:
                rp = pathlib.Path(fname).resolve()
            except OSError:
                continue
            if rp == hot:
                executed = True
                hot_tt += tt
    return hot_tt, total, executed


def render_stats(stats: Any, view: str, rows: int, root: pathlib.Path) -> str:
    """The flat or callers view by self time, with the tree's path stripped."""
    import io

    out = io.StringIO()
    stats.stream = out
    stats.sort_stats("tottime")
    if view == "flat":
        stats.print_stats(rows)
    else:
        stats.print_callers(rows)
    return out.getvalue().replace(str(root) + "/", "")


def _fail(**record: Any) -> int:
    print(json.dumps(record))
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="source tree to import from")
    ap.add_argument("--package", required=True, help="import name of the package under test")
    ap.add_argument(
        "--alias", required=True, help="name the package is bound to for setup and call"
    )
    ap.add_argument(
        "--package-root", required=True, help="directory under root put on sys.path, '.' for flat"
    )
    ap.add_argument(
        "--setup", required=True, help="statements run once, with the alias, ROOT and SEED bound"
    )
    ap.add_argument("--call", required=True, help="expression to time, in the setup namespace")
    ap.add_argument("--hot", required=True, help="the hot file, relative to root")
    ap.add_argument(
        "--seed", type=int, required=True, help="the instance of the input, bound as SEED"
    )
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--label", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--flat-rows", type=int, default=14)
    ap.add_argument("--caller-rows", type=int, default=10)
    ap.add_argument("--no-counters", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    hot = (root / args.hot).resolve()
    pyc_fresh = _pyc_fresh(hot) if hot.exists() else False

    sys.path.insert(0, str((root / args.package_root).resolve()))
    # The clock is bound here, before the package under test is imported, and
    # held in a local: a patch that replaces time.perf_counter at import, or
    # reaches into this module through sys.modules, does not reach this one.
    now = time.perf_counter
    t0 = now()
    module = importlib.import_module(args.package)
    import_s = now() - t0
    package_file = pathlib.Path(str(module.__file__)).resolve()
    if root not in package_file.parents:
        return _fail(
            error="package resolved outside the requested tree",
            package_file=str(package_file),
            root=str(root),
        )

    def fresh_scope() -> dict[str, Any]:
        return {args.alias: module, "ROOT": root, "SEED": args.seed}

    scope = fresh_scope()
    t0 = now()
    exec(args.setup, scope)
    setup_s = now() - t0

    base: dict[str, Any] = {
        "label": args.label,
        "root": str(root),
        "package_file": str(package_file),
        "pyc_fresh_before_import": pyc_fresh,
        "pid": os.getpid(),
        "hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "seed": args.seed,
        "python": list(sys.version_info[:3]),
        "import_s": import_s,
        "setup_s": setup_s,
    }

    def fp_of(result: Any) -> str:
        if hasattr(result, "__next__"):
            raise UnfingerprintableError(
                "the call returned an iterator, so nothing was computed inside the "
                "timed region; wrap the call in list() or otherwise consume it"
            )
        return fingerprint(result)

    if args.verify:
        import cProfile
        import pstats

        base["fixed_s"] = process_age_s()
        profiler = cProfile.Profile()
        t0 = now()
        profiler.enable()
        try:
            result = eval(args.call, scope)
            materialize(result)
        except UnfingerprintableError as e:
            profiler.disable()
            return _fail(error=str(e), **base)
        profiler.disable()
        call_s = now() - t0
        stats = pstats.Stats(profiler)
        hot_tottime, total_tt, executed = hot_self_time(stats, hot)
        try:
            result_fp = fp_of(result)
        except UnfingerprintableError as e:
            return _fail(error=str(e), **base)
        base.update(
            kind="verify",
            hot_executed=executed,
            hot_tottime_share=hot_tottime / total_tt if total_tt else 0.0,
            call_s=call_s,
            result_fp=result_fp,
        )
        print(json.dumps(base))
        return 0

    if args.profile:
        import cProfile
        import pstats

        base["fixed_s"] = process_age_s()
        plain: list[float] = []
        try:
            for _ in range(PROFILE_PLAIN_RUNS):
                gc.collect()
                t0 = now()
                result = eval(args.call, scope)
                materialize(result)
                plain.append(now() - t0)
            gc.collect()
            profiler = cProfile.Profile()
            profiler.enable()
            materialize(eval(args.call, scope))
            profiler.disable()
        except UnfingerprintableError as e:
            return _fail(error=str(e), **base)
        stats = pstats.Stats(profiler)
        hot_tottime, total_tt, executed = hot_self_time(stats, hot)
        try:
            result_fp = fp_of(result)
        except UnfingerprintableError as e:
            return _fail(error=str(e), **base)
        base.update(
            kind="profile",
            call_s=min(plain),
            hot_executed=executed,
            hot_s=hot_tottime,
            total_s=total_tt,
            hot_tottime_share=hot_tottime / total_tt if total_tt else 0.0,
            flat=render_stats(stats, "flat", args.flat_rows, root),
            callers=render_stats(stats, "callers", args.caller_rows, root),
            result_fp=result_fp,
        )
        print(json.dumps(base))
        return 0

    prov = None if args.no_counters else _load_provenance()

    gc.freeze()
    samples: list[dict[str, Any]] = []
    result = None
    base["fixed_s"] = process_age_s()
    for repeat in range(args.repeats):
        # Every timed call gets a freshly built input. A result cached on the
        # input object (t1_p2_w16 stored the whole answer in the graph's own
        # cache dict and recorded 500x to 6400x) then never hits. Setup runs
        # outside the timed region, as before.
        if repeat:
            scope = fresh_scope()
            exec(args.setup, scope)
        gc.collect()
        before = prov.counters() if prov is not None else None
        t0 = now()
        result = eval(args.call, scope)
        t1 = now()
        # The walk is inside the region: pyparsing_p3_w16 from round 14 returned
        # lazy lists that split their text when first read, after the clock.
        try:
            materialize(result)
        except UnfingerprintableError as e:
            return _fail(error=str(e), **base)
        t2 = now()
        sample = {"t": t2 - t0, "walk_s": t2 - t1, "contaminated": False, "reasons": []}
        if prov is not None and before is not None:
            d = prov.delta(before, prov.counters())
            bad, reasons = prov.is_contaminated(d)
            sample.update(contaminated=bad, reasons=reasons)
        samples.append(sample)

    try:
        result_fp = fp_of(result)
    except UnfingerprintableError as e:
        return _fail(error=str(e), **base)
    clean: list[float] = [float(s["t"]) for s in samples if not s["contaminated"]]
    times = [float(s["t"]) for s in samples]
    base.update(
        kind="run",
        repeats=args.repeats,
        samples=samples,
        min_clean=min(clean) if clean else None,
        min_all=min(times),
        # The first call and the best of the rest, kept apart: a cache keyed on
        # the input's content survives the fresh setup, and shows as later
        # calls far faster than the first. The referee compares the gap on the
        # two trees.
        first_s=times[0],
        warm_s=min(times[1:]) if len(times) > 1 else None,
        walk_s=min(float(s["walk_s"]) for s in samples),
        n_contaminated=len(samples) - len(clean),
        result_fp=result_fp,
    )
    print(json.dumps(base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
