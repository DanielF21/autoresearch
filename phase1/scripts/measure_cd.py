"""Measurements C and D, in parallel on three boxes.

  box c0: measurement C, null comparison with two unmodified checkouts
  box c1: measurement C, whitespace patch in the hot file
  box d : measurement D, cost of instruction counting

Each comparison is self contained inside one box, which measurement B showed is
the condition under which boxes can run side by side safely.

Usage: uv run python scripts/measure_cd.py [--quartets 12] [--m 7] [--smoke]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import statistics
import sys
import threading

import sail

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

INBOX = pathlib.Path(__file__).parent.parent / "inbox"
FILES = ["provenance.py", "workload.py", "bench_worker.py", "null_worker.py",
         "drive_c.py", "drive_d.py"]
REMOTE = "/workspace/harness"
OUT = "/workspace/out.jsonl"


def mad(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return 1.4826 * statistics.median([abs(x - m) for x in xs])


def boot_median(xs, q, B=10000, seed=1):
    rng = random.Random(seed)
    n = len(xs)
    meds = sorted(statistics.median([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return meds[min(B - 1, max(0, int(q * B)))]


def parse(text):
    out = []
    for line in (text or "").splitlines():
        if line.strip().startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def job(sb, cmd, key, results, timeout):
    try:
        r = sb.run(cmd, timeout=timeout)
        results[key + "_rc"] = r.exit_code
        results[key + "_stdout"] = (r.stdout or "")[-3000:]
        if r.exit_code:
            results[key + "_stderr"] = (r.stderr or "")[-3000:]
    except Exception as e:  # noqa: BLE001
        results[key + "_exc"] = repr(e)
    finally:
        try:
            results[key] = sb.fs.read(OUT).decode()
        except Exception as e:  # noqa: BLE001
            results[key + "_readerr"] = repr(e)


def analyse_c(recs, mode):
    rep = {"mode": mode}
    fatal = [r for r in recs if r.get("kind") == "fatal"]
    if fatal:
        rep["fatal"] = fatal
        return rep
    setup = next((r for r in recs if r.get("kind") == "setup"), {})
    rep["patch_line"] = setup.get("patch_inserted_at_line")
    rep["verify"] = [{k: v.get(k) for k in ("tree", "bench", "hot_executed", "hot_tottime_share",
                                            "result_fp", "edges")}
                     for v in recs if v.get("kind") == "verify"]
    L = [r for r in recs if r.get("kind") == "launch" and not r.get("error")]
    rep["launch_errors"] = [r for r in recs if r.get("kind") == "launch" and r.get("error")][:3]
    for bench in sorted({r["bench"] for r in L}):
        rows = [r for r in L if r["bench"] == bench]
        ratios = []
        for q in sorted({r["q"] for r in rows}):
            qr = sorted([r for r in rows if r["q"] == q], key=lambda r: r["pos"])
            if len(qr) != 4:
                continue
            for i, j in ((0, 1), (2, 3)):
                x, y = qr[i], qr[j]
                ta = (x if x["arm"] == "A" else y)
                tb = (y if x["arm"] == "A" else x)
                ra = ta["min_clean"] if ta["min_clean"] is not None else ta["min_all"]
                rb = tb["min_clean"] if tb["min_clean"] is not None else tb["min_all"]
                ratios.append(ra / rb)
        if not ratios:
            continue
        s_rel = mad(ratios)
        up = boot_median(ratios, 0.95) if len(ratios) >= 3 else float("nan")
        lo = boot_median(ratios, 0.05) if len(ratios) >= 3 else float("nan")
        fps = {r["arm"]: set() for r in rows}
        for r in rows:
            fps[r["arm"]].add(r.get("result_fp"))
        total = sum(r["repeats"] for r in rows)
        excl = sum(r.get("n_contaminated", 0) for r in rows)
        rep[bench] = {
            "n_pairs": len(ratios),
            "median_ratio": statistics.median(ratios),
            "s_rel_pct": 100 * s_rel,
            "worst_false_speedup_pct": 100 * (max(ratios) - 1),
            "boot90_ci": [lo, up],
            "noise_floor_ratio": max(1.02, up, 1 + 3 * s_rel),
            "noise_floor_statistical_only": max(up, 1 + 3 * s_rel),
            "bias_halt": abs(statistics.median(ratios) - 1) > 0.01,
            "results_identical": len(fps.get("A", set()) | fps.get("B", set())) == 1,
            "pyc_fresh_all": all(r.get("pyc_fresh_before_import") for r in rows),
            "body_ms": 1e3 * statistics.median([r["min_all"] for r in rows]),
            "launch_wall_s": statistics.median([r["wall_s"] for r in rows]),
            "exclusion_pct": 100 * excl / total if total else 0.0,
        }
    can = [r["rec"]["min_all"] for r in recs if r.get("kind") == "canary" and "min_all" in r.get("rec", {})]
    if can:
        rep["canary"] = {"first": can[0], "last": can[-1], "drift_pct": 100 * (can[-1] / can[0] - 1),
                         "range_pct": 100 * (max(can) / min(can) - 1)}
    return rep


def analyse_d(recs):
    rep = {}
    fatal = [r for r in recs if r.get("kind") == "fatal"]
    if fatal:
        return {"fatal": fatal}
    R = [r for r in recs if r.get("kind") == "d_run"]
    for bench in sorted({r["bench"] for r in R}):
        get = {(r["cfg"], r["reps"]): r for r in R if r["bench"] == bench}
        p1, p2 = get.get(("plain", 1)), get.get(("plain", 2))
        if not (p1 and p2):
            continue
        body_plain = p2["wall_s"] - p1["wall_s"]
        rows = {}
        for cfg in ("plain", "nulgrind", "cg_nosim", "cg_sim"):
            a, b = get.get((cfg, 1)), get.get((cfg, 2))
            if not (a and b):
                continue
            body = b["wall_s"] - a["wall_s"]
            ir1, ir2 = a["counts"].get("I refs"), b["counts"].get("I refs")
            row = {"wall_1x_s": a["wall_s"], "body_s": body, "fixed_s": a["wall_s"] - body,
                   "body_slowdown_x": body / body_plain if body_plain > 0 else float("nan"),
                   "inner_body_slowdown_x": (a["inner_min_s"] / p1["inner_min_s"]
                                             if a.get("inner_min_s") and p1.get("inner_min_s") else None),
                   "result_fp": a.get("result_fp"), "rc": a["rc"], "err": a.get("err", "")[:300]}
            if ir1 and ir2:
                row["Ir_body"] = ir2 - ir1
                row["Ir_fixed"] = ir1 - (ir2 - ir1)
            if cfg == "cg_sim":
                for k in ("D1 misses", "LL misses", "D refs"):
                    if a["counts"].get(k) is not None and b["counts"].get(k) is not None:
                        row[f"{k}_body"] = b["counts"][k] - a["counts"][k]
            rows[cfg] = row
        rep[bench] = rows
    det = [r["counts"].get("I refs") for r in recs if r.get("kind") == "d_det"]
    base = next((r["counts"].get("I refs") for r in R
                 if r["bench"] == "short" and r["cfg"] == "cg_nosim" and r["reps"] == 1), None)
    if det and base:
        allv = [base] + det
        rep["determinism"] = {"Ir_runs": allv,
                              "spread_pct": 100 * (max(allv) - min(allv)) / statistics.mean(allv)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quartets", type=int, default=12)
    ap.add_argument("--m", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    require("SAIL_API_KEY")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    image = (sail.Image.debian_amd64
             .apt_install("git", "curl", "build-essential", "valgrind", "time")
             .pip_install("numpy", "scipy", "pandas"))
    app = sail.App.find("autoresearch", mint_if_missing=True)
    names = {"c0": f"measure-c0-{ts}", "c1": f"measure-c1-{ts}", "d": f"measure-d-{ts}"}
    boxes, errs = {}, {}

    def make(key):
        try:
            boxes[key] = sail.Sailbox.create(app=app, name=names[key], image=image, size="m",
                                             disk_limit_gib=32, timeout=10800)
        except Exception as e:  # noqa: BLE001
            errs[key] = repr(e)

    print(f"creating {', '.join(names.values())}...", flush=True)
    ths = [threading.Thread(target=make, args=(k,)) for k in names]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    results: dict = {}
    try:
        if errs:
            raise RuntimeError(f"box creation failed: {errs}")
        for sb in boxes.values():
            for f in FILES:
                sb.fs.write(f"{REMOTE}/{f}", (INBOX / f).read_bytes())
        q, m = (1, 1) if args.smoke else (args.quartets, args.m)
        cmds = {
            "c0": f"cd {REMOTE} && python3 drive_c.py --mode C0 --quartets {q} --m {m} --out {OUT}",
            "c1": f"cd {REMOTE} && python3 drive_c.py --mode C1 --quartets {q} --m {m} --out {OUT}",
            "d": f"cd {REMOTE} && python3 drive_d.py {'--smoke' if args.smoke else ''} --out {OUT}",
        }
        print("all three running in parallel...", flush=True)
        ths = [threading.Thread(target=job, args=(boxes[k], cmds[k], k, results, 7000))
               for k in cmds]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    finally:
        for k, sb in boxes.items():
            try:
                sb.terminate()
            except Exception as e:  # noqa: BLE001
                print(f"!! failed to terminate {names[k]}: {e!r}", file=sys.stderr)
        print("all boxes terminated", flush=True)

    for key in ("c0", "c1", "d"):
        for suffix in ("_rc", "_exc", "_stderr", "_readerr"):
            if results.get(key + suffix) not in (None, 0, ""):
                print(f"!! {key}{suffix}: {str(results[key + suffix])[-600:]}")

    outdir = pathlib.Path("runs/measure-cd")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{ts}{'-smoke' if args.smoke else ''}"
    for key in ("c0", "c1", "d"):
        (outdir / f"{tag}-{key}.jsonl").write_text(results.get(key, ""))

    report = {"c0": analyse_c(parse(results.get("c0")), "C0"),
              "c1": analyse_c(parse(results.get("c1")), "C1"),
              "d": analyse_d(parse(results.get("d")))}
    (outdir / f"{tag}.report.json").write_text(json.dumps(report, indent=2, default=str))

    for key in ("c0", "c1"):
        rc = report[key]
        print(f"\n=== {rc['mode']} ===")
        if rc.get("fatal"):
            print(f"  FATAL: {json.dumps(rc['fatal'])[:800]}")
            continue
        for v in rc.get("verify", []):
            print(f"  verify {v['tree'][-4:]} {v['bench']}: hot_executed={v['hot_executed']} "
                  f"hot_share={v['hot_tottime_share']:.2f} fp={v['result_fp']} edges={v['edges']}")
        for bench in ("long", "short"):
            b = rc.get(bench)
            if not b:
                continue
            print(f"  {bench}: body {b['body_ms']:.1f}ms, launch {b['launch_wall_s']:.1f}s, "
                  f"{b['n_pairs']} pairs")
            print(f"    median ratio {b['median_ratio']:.4f}  s_rel {b['s_rel_pct']:.2f}%  "
                  f"worst false speedup {b['worst_false_speedup_pct']:+.2f}%")
            print(f"    90% CI of median [{b['boot90_ci'][0]:.4f}, {b['boot90_ci'][1]:.4f}]  "
                  f"noise floor {b['noise_floor_ratio']:.4f} (statistical {b['noise_floor_statistical_only']:.4f})")
            print(f"    bias_halt={b['bias_halt']} results_identical={b['results_identical']} "
                  f"pyc_fresh_all={b['pyc_fresh_all']} excluded={b['exclusion_pct']:.2f}%")
        if rc.get("canary"):
            c = rc["canary"]
            print(f"  canary drift {c['drift_pct']:+.2f}% range {c['range_pct']:.2f}%")

    d = report["d"]
    print("\n=== D: cost of instruction counting ===")
    if d.get("fatal"):
        print(f"  FATAL: {json.dumps(d['fatal'])[:800]}")
    for bench in ("short", "long"):
        rows = d.get(bench)
        if not rows:
            continue
        print(f"  {bench}:")
        for cfg, r in rows.items():
            ir = f" Ir/body={r['Ir_body']:,} Ir_fixed={r['Ir_fixed']:,}" if "Ir_body" in r else ""
            ms = ""
            if "D1 misses_body" in r:
                ms = f" D1miss/body={r['D1 misses_body']:,} LLmiss/body={r.get('LL misses_body', 0):,}"
            print(f"    {cfg:9s} 1x={r['wall_1x_s']:7.1f}s body={r['body_s']:7.2f}s "
                  f"fixed={r['fixed_s']:6.1f}s slowdown={r['body_slowdown_x']:5.1f}x "
                  f"fp={r['result_fp']}{ir}{ms}")
            if r["rc"]:
                print(f"      rc={r['rc']} {r['err']}")
    if d.get("determinism"):
        print(f"  determinism: Ir spread {d['determinism']['spread_pct']:.5f}% over "
              f"{len(d['determinism']['Ir_runs'])} runs")
    print(f"\nwrote {outdir}/{tag}-*.jsonl and {tag}.report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
