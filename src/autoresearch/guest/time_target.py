"""One launch of the target against one source tree. Prints one JSON object.

The tree is imported by path, never installed, so two trees can be timed in
turn from the same box. The launch refuses to run if the package resolves
outside the requested tree.

``--graph`` and ``--call`` are Python expressions from the run config. They are
evaluated with the package bound to ``nx`` and the graph bound to ``G``, which
is how the networkx benchmark suite expresses its own bodies.

Modes:

- run (default): ``--repeats`` timed calls, each wrapped in the provenance
  counters so a sample that saw steal, throttling, a page fault, a CPU
  migration or memory pressure is marked contaminated.
- ``--verify``: profile one call and report whether the hot file executed and
  what share of self time it took. Also warms the ``.pyc`` so no later launch
  pays compilation.
- ``--calls N``: call the target N times with no timing and no counters. Used
  under cachegrind, where N = 1 and N = 2 isolate the per call instruction
  count from the fixed startup cost.
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
from typing import Any

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))


def fingerprint(result: Any) -> str:
    """A short hash of the result so both trees can be shown to compute the same thing."""
    try:
        summary = repr(round(sum(result.values()), 9))
    except (AttributeError, TypeError):
        summary = repr(result)
    return hashlib.md5(summary.encode()).hexdigest()[:10]


def _load_provenance() -> Any:
    """The sibling provenance module, found through this file's own directory."""
    import provenance  # type: ignore[import-not-found]

    return provenance


def _pyc_fresh(hot: pathlib.Path) -> bool:
    pyc = pathlib.Path(importlib.util.cache_from_source(str(hot)))
    return pyc.exists() and pyc.stat().st_mtime >= hot.stat().st_mtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="source tree to import from")
    ap.add_argument("--package", default="networkx")
    ap.add_argument("--graph", required=True, help="expression building G, with nx bound")
    ap.add_argument("--call", required=True, help="expression to time, with nx and G bound")
    ap.add_argument("--hot", default="networkx/algorithms/cluster.py")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--label", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--calls", type=int, default=0)
    ap.add_argument("--no-counters", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    hot = (root / args.hot).resolve()
    pyc_fresh = _pyc_fresh(hot) if hot.exists() else False

    sys.path.insert(0, str(root))
    nx = importlib.import_module(args.package)
    nx_file = pathlib.Path(str(nx.__file__)).resolve()
    if root not in nx_file.parents:
        print(
            json.dumps(
                {
                    "error": "package resolved outside the requested tree",
                    "package_file": str(nx_file),
                    "root": str(root),
                }
            )
        )
        return 3

    graph = eval(args.graph, {"nx": nx})
    base: dict[str, Any] = {
        "label": args.label,
        "root": str(root),
        "package_file": str(nx_file),
        "pyc_fresh_before_import": pyc_fresh,
        "pid": os.getpid(),
        "hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
    }
    scope = {"nx": nx, "G": graph}

    if args.calls:
        result = None
        for _ in range(args.calls):
            result = eval(args.call, scope)
        base.update(kind="calls", calls=args.calls, result_fp=fingerprint(result))
        print(json.dumps(base))
        return 0

    if args.verify:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        result = eval(args.call, scope)
        profiler.disable()
        stats = pstats.Stats(profiler)
        files: set[pathlib.Path] = set()
        hot_tottime = 0.0
        total_tt = 0.0
        for (fname, _line, _fn), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
            total_tt += tt
            if fname and not fname.startswith(("<", "~")):
                try:
                    rp = pathlib.Path(fname).resolve()
                except OSError:
                    continue
                files.add(rp)
                if rp == hot:
                    hot_tottime += tt
        base.update(
            kind="verify",
            hot_executed=hot in files,
            hot_tottime_share=hot_tottime / total_tt if total_tt else 0.0,
            result_fp=fingerprint(result),
        )
        print(json.dumps(base))
        return 0

    prov = None if args.no_counters else _load_provenance()

    gc.freeze()
    samples: list[dict[str, Any]] = []
    result = None
    for _ in range(args.repeats):
        gc.collect()
        before = prov.counters() if prov is not None else None
        t0 = time.perf_counter()
        result = eval(args.call, scope)
        t1 = time.perf_counter()
        if prov is not None and before is not None:
            d = prov.delta(before, prov.counters())
            bad, reasons = prov.is_contaminated(d)
            samples.append({"t": t1 - t0, "contaminated": bad, "reasons": reasons})
        else:
            samples.append({"t": t1 - t0, "contaminated": False, "reasons": []})

    clean: list[float] = [float(s["t"]) for s in samples if not s["contaminated"]]
    base.update(
        kind="run",
        repeats=args.repeats,
        samples=samples,
        min_clean=min(clean) if clean else None,
        min_all=min(float(s["t"]) for s in samples),
        n_contaminated=len(samples) - len(clean),
        result_fp=fingerprint(result),
    )
    print(json.dumps(base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
