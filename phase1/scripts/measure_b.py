"""Measurement B: contended timing noise.

Does load elsewhere corrupt the referee's timings, and can it be detected per
run when it happens?

Arms (B2a deleted: Phase 0 found smt/active=0, so there is no sibling to contend):
  B0    idle
  B2c   3 concurrent in-box workers
  B1    3 other sailboxes saturated with memory-heavy load

ABBA within each cycle, order flipped on alternate cycles. Measurement A found
the box speeds up ~8.7% monotonically over 20 minutes, so a block design would
confound the arm with drift completely.

Usage: uv run python scripts/measure_b.py [--k 6] [--cycles 3]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys
import threading
import time

import sail

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

INBOX = pathlib.Path(__file__).parent.parent / "inbox"
FILES = ["provenance.py", "workload.py", "bench_worker.py", "run_arm.py"]
REMOTE = "/workspace/harness"

# Saturate all vCPUs with a memory-bandwidth-heavy workload, not a spin loop.
XBOX_LOAD = ("nohup stress-ng --vm 4 --vm-bytes 1G --timeout 3600s "
             ">/dev/null 2>&1 & echo started")


def mad_abs(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics.median(xs)
    return 1.4826 * statistics.median([abs(x - m) for x in xs])


def parse_jsonl(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="launches per arm instance (2 per cycle -> 12/arm/cycle)")
    ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--pin", type=int, default=2)
    ap.add_argument("--load-boxes", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    require("SAIL_API_KEY")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    image = (sail.Image.debian_amd64
             .apt_install("git", "curl", "build-essential", "valgrind", "time", "stress-ng")
             .pip_install("numpy"))
    app = sail.App.find("autoresearch", mint_if_missing=True)

    print(f"creating referee + {args.load_boxes} load boxes...", flush=True)
    ref = sail.Sailbox.create(app=app, name=f"measure-b-ref-{ts}", image=image,
                              size="m", disk_limit_gib=32, timeout=10800)
    loads = []
    for i in range(args.load_boxes):
        loads.append(sail.Sailbox.create(app=app, name=f"measure-b-load{i}-{ts}", image=image,
                                         size="m", disk_limit_gib=32, timeout=10800))
    print(f"  referee={ref.vcpu_count}vcpu, {len(loads)} load boxes up", flush=True)

    records: list[dict] = []
    try:
        for f in FILES:
            ref.fs.write(f"{REMOTE}/{f}", (INBOX / f).read_bytes())

        base = f"cd {REMOTE} && python3 run_arm.py"
        r = ref.run(f"{base} --mode facts", timeout=600)
        records += parse_jsonl(r.stdout or "")
        facts = next((x["facts"] for x in records if x.get("kind") == "static_facts"), {})
        print(f"  flag_hash={facts.get('cpu',{}).get('flag_hash')} "
              f"nproc={facts.get('cpu',{}).get('nproc')}", flush=True)

        r = ref.run(f"{base} --mode calibrate --pin {args.pin}", timeout=600)
        cal = parse_jsonl(r.stdout or "")
        records += cal
        n_anchor = next(x["n_anchor"] for x in cal if x.get("kind") == "calibration")
        print(f"  anchor n={n_anchor}", flush=True)

        xbox_on = False

        def set_xbox(on: bool) -> None:
            """Start or stop cross-box load, and VERIFY it actually happened.

            An unverified B1 arm is worthless: if stress-ng silently fails to
            launch, "no cross-box effect detected" is measuring nothing and
            would be the most misleading possible result.
            """
            nonlocal xbox_on
            if on == xbox_on:
                return
            def act(b):
                b.run(XBOX_LOAD if on else "pkill -f stress-ng || true", timeout=120)
            ths = [threading.Thread(target=act, args=(b,)) for b in loads]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            time.sleep(8)  # let load ramp and loadavg register
            # `pgrep -c` prints "0" AND exits 1 when it finds nothing, so a
            # `|| echo 0` fallback double-counts. Emit tagged lines and parse
            # defensively rather than relying on field position.
            probe = ("printf 'NP=%s\\n' \"$(pgrep -c stress-ng 2>/dev/null | tail -1)\"; "
                     "printf 'LA=%s\\n' \"$(cut -d' ' -f1 /proc/loadavg)\"")
            checks = []
            for b in loads:
                v = b.run(probe, timeout=60)
                np_, la_ = 0, 0.0
                for line in (v.stdout or "").splitlines():
                    line = line.strip()
                    if line.startswith("NP="):
                        try:
                            np_ = int(line[3:].strip() or 0)
                        except ValueError:
                            np_ = 0
                    elif line.startswith("LA="):
                        try:
                            la_ = float(line[3:].strip() or 0)
                        except ValueError:
                            la_ = 0.0
                checks.append((np_, la_))
            live = sum(1 for n, _ in checks if n > 0)
            records.append({"kind": "xbox_state", "on": on, "checks": checks, "live_boxes": live})
            if on and live < len(loads):
                raise RuntimeError(
                    f"cross-box load did not start on all boxes: {checks}. "
                    f"B1 would measure nothing. Aborting rather than reporting a false null.")
            if not on and live > 0:
                print(f"  !! warning: stress-ng still running on {live} boxes after stop",
                      file=sys.stderr)
            print(f"    xbox load {'ON' if on else 'OFF'}: {checks}", flush=True)
            xbox_on = on

        arms = ["B0", "B2c", "B1"]
        for cycle in range(args.cycles):
            order = arms + arms[::-1]          # ABBA: cancels a linear drift term
            if cycle % 2:
                order = order[::-1]
            print(f"[cycle {cycle}] order = {' '.join(order)}", flush=True)
            for arm in order:
                set_xbox(arm == "B1")
                cmd = (f"{base} --mode arm --arm {arm} --k {args.k} --m {args.m} "
                       f"--n {n_anchor} --pin {args.pin} --cycle {cycle}")
                r = ref.run(cmd, timeout=3600)
                recs = parse_jsonl(r.stdout or "")
                records += recs
                summ = next((x for x in recs if x.get("kind") == "arm_summary"), {})
                can = next((x for x in recs if x.get("kind") == "canary"), {})
                print(f"  {arm}: {summ.get('wall_s',0):.0f}s "
                      f"steal_delta={summ.get('steal_delta')} "
                      f"psi={summ.get('psi_after')} "
                      f"canary={can.get('rec',{}).get('min_all',float('nan')):.4f}s", flush=True)
        set_xbox(False)
    finally:
        for b in loads:
            try:
                b.terminate()
            except Exception as e:  # noqa: BLE001
                print(f"  !! failed to terminate {b.app_name}: {e!r}", file=sys.stderr)
        if args.keep:
            print(f"referee measure-b-ref-{ts} left running")
        else:
            ref.terminate()
            print("referee terminated")

    outdir = pathlib.Path("runs/measure-b")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{ts}.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    # analysis: pair every arm against the B0 measured in the SAME cycle
    L = [r for r in records if r.get("kind") == "launch" and "min_all" in r]
    report = {"arms": {}, "cycles": {}}
    for cyc in sorted({r["cycle"] for r in L}):
        base_rows = [r["min_clean"] or r["min_all"] for r in L
                     if r["cycle"] == cyc and r["arm"] == "B0"]
        if not base_rows:
            continue
        b0 = statistics.median(base_rows)
        b0_spread = 100 * mad_abs(base_rows) / b0
        report["cycles"][cyc] = {"B0_median_ms": b0 * 1e3, "B0_s_rel_pct": b0_spread}
        for arm in sorted({r["arm"] for r in L if r["cycle"] == cyc}):
            xs = [r["min_clean"] or r["min_all"] for r in L
                  if r["cycle"] == cyc and r["arm"] == arm]
            T = statistics.median(xs)
            report["arms"].setdefault(arm, []).append({
                "cycle": cyc,
                "median_ms": T * 1e3,
                "inflation_pct": 100 * (T / b0 - 1),
                "s_rel_pct": 100 * mad_abs(xs) / T,
                "spread_ratio_vs_B0": (100 * mad_abs(xs) / T) / b0_spread if b0_spread else float("nan"),
            })

    steal = {}
    for r in records:
        if r.get("kind") == "arm_summary":
            steal.setdefault(r["arm"], []).append(r.get("steal_delta", 0))
    report["steal_by_arm"] = steal

    print("\n=== inflation vs same-cycle B0 ===")
    for arm, rows in report["arms"].items():
        inf = statistics.median([x["inflation_pct"] for x in rows])
        sr = statistics.median([x["spread_ratio_vs_B0"] for x in rows])
        print(f"  {arm}: median inflation {inf:+.2f}%   spread ratio vs B0 {sr:.2f}x   "
              f"steal_deltas={steal.get(arm)}")

    b1 = report["arms"].get("B1", [])
    b2c = report["arms"].get("B2c", [])
    b1_inf = statistics.median([x["inflation_pct"] for x in b1]) if b1 else float("nan")
    b1_steal = sum(steal.get("B1", []))
    b2c_inf = statistics.median([x["inflation_pct"] for x in b2c]) if b2c else float("nan")
    b2c_sr = statistics.median([x["spread_ratio_vs_B0"] for x in b2c]) if b2c else float("nan")

    # Steal must be judged against the IDLE arm, not against zero. Background
    # steal of a few jiffies appears on every arm including B0, and on this
    # platform B0 had MORE total steal than B1. Absolute steal is not a signal.
    b0_steal = sum(steal.get("B0", []))
    steal_excess = b1_steal - b0_steal
    lines = []
    if steal_excess > 0:
        lines.append(f"Steal excess over idle: {steal_excess} jiffies.")
    else:
        lines.append(f"Steal is NOT a usable contention detector here: B1={b1_steal} vs "
                     f"idle B0={b0_steal} jiffies. The planned production guard is blind to "
                     f"an effect the canary detects easily. Use the canary instead.")
    if abs(b1_inf) > 2:
        lines.append("CROSS-BOX EFFECT DETECTED. Boxes share an oversubscribed host. Record "
                     "steal on every production run, auto-invalidate contaminated candidates. "
                     "Referee throughput does NOT scale linearly with box count.")
    else:
        lines.append("No cross-box effect detected at this load level. Not proof of isolation "
                     "(placement is not observable), but N referee boxes should give ~N times "
                     "throughput. THE 10-HOUR SERIAL CONSTRAINT LIKELY DISSOLVES.")
    if b2c_inf > 3 or b2c_sr > 2:
        lines.append("In-box load corrupts referee timings. Referee needs a dedicated box and "
                     "the cachegrind fan-out gets its own.")
    else:
        lines.append("In-box load is tolerable, but a dedicated referee box costs ~$2 and "
                     "remains the default.")
    report["verdict"] = lines
    print("\n=== VERDICT ===")
    for line in lines:
        print(f"  {line}")

    (outdir / f"{ts}.report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outdir}/{ts}.jsonl and .report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
