"""Measurement A: uncontended timing noise.

Uploads the in-box harness, runs it, fetches raw JSONL, analyses it against the
decision rule fixed in advance.

Usage: uv run python scripts/measure_a.py [--k 30] [--sets 3] [--keep]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys

import sail

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

INBOX = pathlib.Path(__file__).parent.parent / "inbox"
FILES = ["provenance.py", "workload.py", "bench_worker.py", "drive_a.py"]
REMOTE = "/workspace/harness"


def mad_abs(xs: list[float]) -> float:
    """1.4826 * MAD. Robust to the one or two host events that occur in any
    30-launch set; a standard deviation would be inflated into uselessness."""
    if len(xs) < 2:
        return 0.0
    med = statistics.median(xs)
    return 1.4826 * statistics.median([abs(x - med) for x in xs])


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def analyse(records: list[dict]) -> dict:
    launches = [r for r in records if r.get("kind") == "launch" and "min_all" in r]
    canaries = [r for r in records if r.get("kind") == "canary"]
    calib = next((r for r in records if r.get("kind") == "calibration"), {})
    facts = next((r["facts"] for r in records if r.get("kind") == "static_facts"), {})

    report: dict = {"anchors": {}, "canary": {}, "provenance": {}, "errors": []}
    report["errors"] = [r for r in records if r.get("error")]

    if facts:
        report["provenance"] = {
            "cpu_model": facts["cpu"]["model_name"],
            "flag_hash": facts["cpu"]["flag_hash"],
            "vuln_hash": facts["mitigations"]["vuln_hash"],
            "python_config_hash": facts["python"]["config_hash"],
            "boot_id": facts["platform"]["boot_id"],
            "l3": next((c["size"] for c in facts["cpu"]["caches"] if c["level"] == "3"), "?"),
            "mitigations": facts["mitigations"]["vulnerabilities"],
        }

    report["canary"] = {
        "n_iters": calib.get("canary_n"),
        "per_set": [c["rec"]["min_all"] for c in canaries if "min_all" in c.get("rec", {})],
    }

    for label in sorted({r["label"] for r in launches}):
        rows = [r for r in launches if r["label"] == label]
        # median of mins: min within a process (additive noise), median across
        # processes (real ASLR / hash-seed variation, not noise)
        mins = [r["min_clean"] if r["min_clean"] is not None else r["min_all"] for r in rows]
        T = statistics.median(mins)
        spread_abs = mad_abs(mins)
        total_samples = sum(r["repeats"] for r in rows)
        excluded = sum(r["n_contaminated"] for r in rows)
        # Pooling across sets conflates DRIFT with NOISE. The within-set spread
        # is the instrument's noise; the between-set movement is drift. They
        # need entirely different responses, so never report only the pooled number.
        per_set, within, adj_ratios = {}, [], []
        for s in sorted({r["set"] for r in rows}):
            sm = [r["min_clean"] or r["min_all"] for r in rows if r["set"] == s]
            per_set[s] = statistics.median(sm)
            if len(sm) > 1:
                within.append(100 * mad_abs(sm) / statistics.median(sm))
                # The referee compares a baseline to a patch timed immediately
                # beside it, never across a 5-minute window. This is that spread.
                adj_ratios += [sm[i] / sm[i + 1] for i in range(len(sm) - 1)]
        report["anchors"][label] = {
            "n_launches": len(rows),
            "T_median_of_mins": T,
            "spread_abs": spread_abs,
            "s_rel_pct": 100 * spread_abs / T if T else float("nan"),
            "p90_excess_pct": 100 * (pct(mins, 0.90) / T - 1) if T else float("nan"),
            "p95_excess_pct": 100 * (pct(mins, 0.95) / T - 1) if T else float("nan"),
            "max_excess_pct": 100 * (max(mins) / T - 1) if T else float("nan"),
            "exclusion_rate_pct": 100 * excluded / total_samples if total_samples else 0.0,
            "s_rel_within_set_pct": statistics.median(within) if within else float("nan"),
            "s_rel_pooled_pct": 100 * spread_abs / T if T else float("nan"),
            "adjacent_pair_mad_pct": 100 * mad_abs(adj_ratios) if len(adj_ratios) > 1 else float("nan"),
            "adjacent_pair_central95_pct": (
                100 * max(abs(pct(adj_ratios, 0.025) - 1), abs(pct(adj_ratios, 0.975) - 1))
                if len(adj_ratios) > 2 else float("nan")),
            "per_set_median": per_set,
            "drift_pct": (100 * (max(per_set.values()) / min(per_set.values()) - 1)
                          if len(per_set) > 1 else 0.0),
        }

    # Two anchors determine spread_abs(T) = a + b*T exactly.
    a1 = report["anchors"].get("a1_50ms")
    a2 = report["anchors"].get("a2_1s")
    if a1 and a2:
        t1, s1 = a1["T_median_of_mins"], a1["spread_abs"]
        t2, s2 = a2["T_median_of_mins"], a2["spread_abs"]
        if t2 != t1:
            b = (s2 - s1) / (t2 - t1)
            a = s1 - b * t1
            # A negative intercept is not physical. It means the two-point fit
            # found the short anchor proportionally LESS noisy than the long one,
            # i.e. spread is essentially purely proportional with no fixed floor.
            # Report the raw fit but clamp what the referee actually uses.
            report["floor_model"] = {
                "abs_floor_us_raw": a * 1e6,
                "abs_floor_us": max(a, 0.0) * 1e6,
                "rel_floor_pct": 100 * b,
                "intercept_negative": a < 0,
                "form": "threshold(T) = max(rel_floor * T, abs_floor)",
            }
    return report


def verdict(report: dict) -> tuple[str, str]:
    a2 = report["anchors"].get("a2_1s")
    if not a2:
        return "UNKNOWN", "1s anchor missing"
    # Judge on the WITHIN-set spread. The pooled number includes drift, and
    # drift is handled architecturally (adjacent paired baselines), not by
    # declaring the platform unusable.
    s_rel, p95 = a2["s_rel_within_set_pct"], a2["p95_excess_pct"]
    if s_rel <= 1.0 and p95 <= 2.0:
        return "PASS", ("s_rel <= 1% and p95 excess <= 2%. Proceed with the wall clock "
                        "referee as architected.")
    if s_rel <= 3.0:
        return "MARGINAL", ("s_rel between 1% and 3%. Every candidate needs >= 5 interleaved "
                            "ABBA pairs against a freshly timed baseline, roughly doubling "
                            "per-candidate cost and forcing the cheap gate to cut the queue.")
    return "FAIL", ("s_rel > 3%. Wall clock is not a usable accept signal on this VM class. "
                    "Retry on a larger size, then demote wall clock to confirmation on a "
                    "shortlist and promote instruction counts to primary.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--pin", type=int, default=2)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--size", default="m")
    args = ap.parse_args()
    require("SAIL_API_KEY")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"measure-a-{ts}"
    image = (sail.Image.debian_amd64
             .apt_install("git", "curl", "build-essential", "valgrind", "time", "stress-ng")
             .pip_install("numpy"))
    app = sail.App.find("autoresearch", mint_if_missing=True)
    print(f"creating {name}...", flush=True)
    sb = sail.Sailbox.create(app=app, name=name, image=image, size=args.size,
                             disk_limit_gib=32, timeout=7200)
    try:
        for f in FILES:
            sb.fs.write(f"{REMOTE}/{f}", (INBOX / f).read_bytes())
        print("harness uploaded; running (this takes ~20 min)...", flush=True)

        cmd = (f"cd {REMOTE} && SAILBOX_NAME={name} python3 drive_a.py "
               f"--k {args.k} --m {args.m} --sets {args.sets} --pin {args.pin} "
               f"--out /workspace/results_a.jsonl")
        r = sb.run(cmd, timeout=7000)
        print(r.stdout or "", flush=True)
        if r.exit_code != 0:
            print(f"!! exit {r.exit_code}\n{(r.stderr or '')[-3000:]}", file=sys.stderr)

        raw = sb.fs.read("/workspace/results_a.jsonl").decode()
    finally:
        if args.keep:
            print(f"box {name} left running")
        else:
            sb.terminate()
            print(f"box {name} terminated")

    outdir = pathlib.Path("runs/measure-a")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{ts}.jsonl").write_text(raw)

    records = [json.loads(l) for l in raw.splitlines() if l.strip()]
    report = analyse(records)
    v, why = verdict(report)
    report["verdict"] = {"result": v, "rationale": why}
    (outdir / f"{ts}.report.json").write_text(json.dumps(report, indent=2))

    p = report["provenance"]
    print("\n=== provenance ===")
    print(f"  cpu={p.get('cpu_model')} L3={p.get('l3')}")
    print(f"  flag_hash={p.get('flag_hash')} vuln_hash={p.get('vuln_hash')} "
          f"py_cfg={p.get('python_config_hash')}")
    print(f"  boot_id={p.get('boot_id')}")
    c = report["canary"]
    print(f"\n=== canary ({c.get('n_iters')} iters, fixed forever) ===")
    print("  per set: " + ", ".join(f"{x:.4f}s" for x in c.get("per_set", [])))
    print("\n=== anchors ===")
    for label, d in report["anchors"].items():
        print(f"  {label}: T={d['T_median_of_mins']*1000:.3f}ms")
        print(f"      s_rel WITHIN-set = {d['s_rel_within_set_pct']:.3f}%   "
              f"(pooled {d['s_rel_pooled_pct']:.3f}% -- includes drift, do not judge on it)")
        print(f"      drift across sets = {d['drift_pct']:.2f}%")
        print(f"      single adjacent pair: MAD={d['adjacent_pair_mad_pct']:.2f}% "
              f"central95=+/-{d['adjacent_pair_central95_pct']:.2f}%")
        print(f"      p90=+{d['p90_excess_pct']:.2f}% p95=+{d['p95_excess_pct']:.2f}% "
              f"max=+{d['max_excess_pct']:.2f}% launches={d['n_launches']} "
              f"excluded={d['exclusion_rate_pct']:.2f}%")
    if "floor_model" in report:
        fm = report["floor_model"]
        print(f"\n=== noise floor model ===\n  {fm['form']}")
        print(f"  abs_floor = {fm['abs_floor_us']:.1f} us    rel_floor = {fm['rel_floor_pct']:.3f}%")
        if fm.get("intercept_negative"):
            print(f"  note: raw intercept {fm['abs_floor_us_raw']:.1f} us is negative and was "
                  f"clamped to 0. Spread is effectively purely proportional here.")
    if report["errors"]:
        print(f"\n!! {len(report['errors'])} failed launches")
    print(f"\n=== VERDICT: {v} ===\n  {why}")
    print(f"\nwrote {outdir}/{ts}.jsonl and .report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
