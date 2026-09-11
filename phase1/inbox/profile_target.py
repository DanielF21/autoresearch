"""Profile the referee target with cProfile. Runs INSIDE the box.

Phase 1 item: profile the one benchmark, dump it, count the tokens. The token
count decides whether a worker can read the profile directly or has to reach it
through recursive sub-queries.

Target: nx.clustering on erdos_renyi_graph(n, p, seed=42, directed=True), the
body of DirectedAlgorithmBenchmarks.time_clustering.

Writes, per graph size: three binary .prof dumps, text renderings (flat by self
time, flat by cumulative time, top 20, callers, callees) with repo-relative
paths, and summary.json with per-function shares and Amdahl ceilings.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import pathlib
import pstats
import statistics
import sys
import sysconfig
import time

BENCHES = {"long": (1000, 0.05), "short": (1000, 0.01)}


def classify(fname: str, root: str, stdlib_dirs: list[str]) -> str:
    if fname == "~":
        return "builtin"          # C functions: cProfile records them with filename "~"
    if fname.startswith(root):
        return "networkx"
    if any(fname.startswith(d) for d in stdlib_dirs):
        return "stdlib"
    if "site-packages" in fname:
        return "third_party"
    return "other"


def label(key, root: str, stdlib_dirs: list[str]) -> str:
    fname, line, func = key
    if fname == "~":
        return func
    fname = fname.replace(root + "/", "")
    for d in stdlib_dirs:
        fname = fname.replace(d + "/", "<stdlib>/")
    return f"{fname}:{line}({func})"


def render(prof: pathlib.Path, view: str, root: str, stdlib_dirs: list[str]) -> tuple[str, str]:
    s = io.StringIO()
    st = pstats.Stats(str(prof), stream=s)
    if view == "flat_tottime":
        st.sort_stats("tottime").print_stats()
    elif view == "flat_cumulative":
        st.sort_stats("cumulative").print_stats()
    elif view == "top20":
        st.sort_stats("tottime").print_stats(20)
    elif view == "callers":
        st.sort_stats("tottime").print_callers()
    elif view == "callees":
        st.sort_stats("tottime").print_callees()
    raw = s.getvalue()
    rel = raw.replace(root + "/", "")
    for d in stdlib_dirs:
        rel = rel.replace(d + "/", "<stdlib>/")
    return raw, rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/nx_a")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="/workspace/profile")
    args = ap.parse_args()

    root = str(pathlib.Path(args.root).resolve())
    sys.path.insert(0, root)
    import networkx as nx

    if not str(pathlib.Path(nx.__file__).resolve()).startswith(root):
        print(json.dumps({"error": "networkx resolved outside root", "file": nx.__file__}))
        return 3

    paths = sysconfig.get_paths()
    stdlib_dirs = sorted({paths["stdlib"], paths["platstdlib"]}, key=len, reverse=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"python": sys.version, "networkx_file": nx.__file__, "benches": {}}

    for bench, (n, p) in BENCHES.items():
        G = nx.erdos_renyi_graph(n, p, seed=42, directed=True)

        # Unprofiled reference, so cProfile's own overhead is known.
        plain = []
        for _ in range(3):
            gc.collect()
            t0 = time.perf_counter()
            nx.clustering(G)
            plain.append(time.perf_counter() - t0)

        runs = []
        for i in range(args.runs):
            gc.collect()
            pr = cProfile.Profile()
            t0 = time.perf_counter()
            pr.enable()
            nx.clustering(G)
            pr.disable()
            wall = time.perf_counter() - t0
            path = out / f"{bench}_run{i}.prof"
            pr.dump_stats(str(path))
            st = pstats.Stats(str(path))
            runs.append({"path": path, "wall": wall, "total_tt": st.total_tt, "stats": st.stats})

        # Share stability across runs, keyed by function.
        shares = [{k: v[2] / r["total_tt"] for k, v in r["stats"].items()} for r in runs]
        base = runs[0]
        rows = []
        for key, (cc, nc, tt, ct, _callers) in base["stats"].items():
            tt_share = tt / base["total_tt"]
            ct_share = ct / base["total_tt"]
            per_run = [s.get(key, 0.0) for s in shares]
            rows.append({
                "function": label(key, root, stdlib_dirs),
                "kind": classify(key[0], root, stdlib_dirs),
                "calls": nc,
                "primitive_calls": cc,
                "tottime_s": tt,
                "cumtime_s": ct,
                "tottime_share": tt_share,
                "cumtime_share": ct_share,
                "tottime_share_min": min(per_run),
                "tottime_share_max": max(per_run),
                # Speedup ceiling if this function's own time went to zero.
                "ceiling_self": 1 / (1 - tt_share) if tt_share < 1 else None,
                # Ceiling if the function and everything it calls went to zero.
                # Meaningless for the entry point, whose cumulative share is ~100%.
                "ceiling_with_callees": 1 / (1 - ct_share) if ct_share < 0.99 else None,
            })
        rows.sort(key=lambda r: -r["tottime_s"])

        texts = {}
        for view in ("flat_tottime", "flat_cumulative", "top20", "callers", "callees"):
            raw, rel = render(base["path"], view, root, stdlib_dirs)
            (out / f"{bench}_{view}.txt").write_text(rel)
            (out / f"{bench}_{view}.rawpaths.txt").write_text(raw)
            texts[view] = {"chars": len(rel), "chars_rawpaths": len(raw), "lines": rel.count("\n")}

        by_kind: dict[str, float] = {}
        for r in rows:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0.0) + r["tottime_share"]

        summary["benches"][bench] = {
            "n": n, "p": p, "edges": G.number_of_edges(),
            "plain_min_s": min(plain),
            "profiled_wall_s": [r["wall"] for r in runs],
            "overhead_x": statistics.median(r["wall"] for r in runs) / min(plain),
            "n_functions": len(base["stats"]),
            "prof_bytes": base["path"].stat().st_size,
            "tottime_share_by_kind": by_kind,
            "texts": texts,
            "rows": rows,
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"ok": True, "files": sorted(x.name for x in out.iterdir())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
