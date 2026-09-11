"""One launch of the real networkx target, run against a chosen source tree.

Target: DirectedAlgorithmBenchmarks.time_clustering, whose body is
`nx.clustering(G)`. The benchmark module is NOT imported: it builds every graph
in the suite at import time and costs 29 seconds to load. Only the target graph
is built here, with the benchmark's own constructor and seed, so the input is
identical to the suite's and identical across every launch.

Prints one JSON object.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="networkx source tree to import from")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--label", default="")
    ap.add_argument("--hot", default="networkx/algorithms/cluster.py")
    ap.add_argument("--verify", action="store_true",
                    help="profile one call and confirm the hot file actually executes")
    ap.add_argument("--no-counters", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    hot = (root / args.hot).resolve()

    # Did THIS launch have to recompile the hot file? Checked before import,
    # because import is what rewrites the .pyc. Recompilation lands in import,
    # outside the timed region, but it is recorded so the claim is checkable.
    pyc = pathlib.Path(importlib.util.cache_from_source(str(hot)))
    pyc_fresh = pyc.exists() and pyc.stat().st_mtime >= hot.stat().st_mtime

    sys.path.insert(0, str(root))
    import networkx as nx

    nx_file = pathlib.Path(nx.__file__).resolve()
    if root not in nx_file.parents:
        print(json.dumps({"error": "networkx resolved outside the requested tree",
                          "networkx_file": str(nx_file), "root": str(root)}))
        return 3

    G = nx.erdos_renyi_graph(args.n, args.p, seed=42, directed=True)
    base = {"label": args.label, "root": str(root), "networkx_file": str(nx_file),
            "n": args.n, "p": args.p, "edges": G.number_of_edges(),
            "pyc_fresh_before_import": pyc_fresh, "pid": os.getpid(),
            "hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>")}

    if args.verify:
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        res = nx.clustering(G)
        pr.disable()
        st = pstats.Stats(pr)
        files = set()
        hot_tottime = 0.0
        total_tt = 0.0
        for (fname, _line, _fn), (_cc, _nc, tt, _ct, _callers) in st.stats.items():
            total_tt += tt
            if fname and not fname.startswith(("<", "~")):
                try:
                    rp = pathlib.Path(fname).resolve()
                except OSError:
                    continue
                files.add(rp)
                if rp == hot:
                    hot_tottime += tt
        base.update(kind="verify", hot_executed=hot in files,
                    hot_tottime_share=hot_tottime / total_tt if total_tt else 0.0,
                    pyc_exists_after_import=pyc.exists(),
                    result_fp=hashlib.md5(repr(round(sum(res.values()), 9)).encode()).hexdigest()[:10])
        print(json.dumps(base))
        return 0

    prov = None
    if not args.no_counters:
        import provenance as prov  # noqa: F811

    gc.freeze()
    samples = []
    res = None
    for _ in range(args.repeats):
        gc.collect()
        c0 = prov.counters() if prov else None
        t0 = time.perf_counter()
        res = nx.clustering(G)
        t1 = time.perf_counter()
        if prov:
            d = prov.delta(c0, prov.counters())
            bad, reasons = prov.is_contaminated(d)
            samples.append({"t": t1 - t0, "contaminated": bad, "reasons": reasons,
                            "nivcsw": d["ru_nivcsw"], "migrated": d["cpu_migrated"]})
        else:
            samples.append({"t": t1 - t0, "contaminated": False, "reasons": []})

    clean = [s["t"] for s in samples if not s["contaminated"]]
    base.update(kind="run", repeats=args.repeats, samples=samples,
                min_clean=min(clean) if clean else None,
                min_all=min(s["t"] for s in samples),
                n_contaminated=len(samples) - len(clean),
                result_fp=hashlib.md5(repr(round(sum(res.values()), 9)).encode()).hexdigest()[:10])
    print(json.dumps(base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
