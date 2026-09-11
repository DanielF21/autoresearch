"""Measurement C orchestrator. Runs INSIDE the box.

Question: how small a speedup can the referee detect on real networkx code?

Two trees of networkx at the same commit, both imported by path switching so
alternating between them costs nothing. Tree A is always unmodified.
  C0: tree B is a second independent checkout, also unmodified. Tests whether
      the pipeline itself shifts results with no diff at all.
  C1: tree B carries a whitespace-only patch in the hot file, inserted above the
      first function so every function's line numbers shift, as a real patch's
      would. Nothing about the computation changes.
Any ratio away from 1.000 is therefore measurement error. Its spread is the floor.

Launches run in ABBA quartets, BAAB on alternate quartets, same hash seed within
a quartet so each pair is paired on seed. A canary runs at the top of each quartet.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import provenance  # noqa: E402
import workload  # noqa: E402

SHA = "c94928ed9489"
URL = "https://github.com/networkx/networkx"
SEEDS = ["0", "1", "2", "3", "4"]
BENCHES = {"long": (1000, 0.05), "short": (1000, 0.01)}
HOT = "networkx/algorithms/cluster.py"
A, B = "/workspace/nx_a", "/workspace/nx_b"


def sh(cmd: str, timeout: int = 900) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def whitespace_patch(root: str) -> int:
    p = pathlib.Path(root) / HOT
    lines = p.read_text().splitlines(keepends=True)
    idx = next(i for i, l in enumerate(lines) if l.startswith(("@", "def ")))
    lines.insert(idx, "\n")
    p.write_text("".join(lines))
    return idx + 1


def worker(root, n, p, repeats, label, seed, pin, verify=False) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    argv = [sys.executable, str(HERE / "null_worker.py"), "--root", root, "--n", str(n),
            "--p", str(p), "--repeats", str(repeats), "--label", label]
    if verify:
        argv.append("--verify")
    if pin is not None:
        argv = ["taskset", "-c", str(pin)] + argv
    t0 = time.monotonic()
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=1800)
    wall = time.monotonic() - t0
    try:
        rec = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        rec = {"error": "unparseable worker output", "stdout": r.stdout[-500:]}
    if r.returncode != 0:
        rec.setdefault("error", f"rc={r.returncode}")
        rec["stderr"] = r.stderr[-1500:]
    rec.update(wall_s=wall, seed=seed)
    return rec


def canary(pin) -> dict:
    env = dict(os.environ, PYTHONHASHSEED="0")
    argv = ["taskset", "-c", str(pin), sys.executable, str(HERE / "bench_worker.py"),
            "--n", str(workload.CANARY_N), "--repeats", "5", "--label", workload.CANARY_VERSION]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"error": r.stderr[-800:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["C0", "C1"], required=True)
    ap.add_argument("--quartets", type=int, default=12, help="2 pairs each; 12 -> 24 pairs")
    ap.add_argument("--m", type=int, default=7)
    ap.add_argument("--pin", type=int, default=2)
    ap.add_argument("--benches", default="long,short")
    ap.add_argument("--out", default="/workspace/out.jsonl")
    args = ap.parse_args()
    benches = args.benches.split(",")

    fh = open(args.out, "w")

    def emit(o: dict) -> None:
        fh.write(json.dumps(o) + "\n")
        fh.flush()

    waited = 0
    while waited < 180:
        psi = provenance.counters()["psi"].get("some_avg10", 0.0)
        busy = subprocess.run("ps aux | grep -c '[e]xt4lazyinit'", shell=True,
                              capture_output=True, text=True).stdout.strip()
        if psi < 0.5 and busy == "0":
            break
        time.sleep(5)
        waited += 5
    emit({"kind": "settle", "waited_s": waited})
    emit({"kind": "static_facts", "facts": provenance.static_facts()})

    t0 = time.time()
    rc, out = sh(f"git clone -q --depth 50 {URL} {A} && cd {A} && git checkout -q {SHA}")
    if rc:
        emit({"kind": "fatal", "stage": "clone A", "out": out})
        return 2
    rc, out = sh(f"git clone -q {A} {B} && cd {B} && git checkout -q {SHA}")
    if rc:
        emit({"kind": "fatal", "stage": "clone B", "out": out})
        return 2
    # Never let an installed networkx shadow the two trees.
    sh("pip uninstall -y networkx")
    patch_line = whitespace_patch(B) if args.mode == "C1" else None
    _, diff = sh(f"cd {B} && git diff --stat && git diff | head -20")
    emit({"kind": "setup", "mode": args.mode, "sha": SHA, "patch_inserted_at_line": patch_line,
          "diff": diff, "setup_s": time.time() - t0})

    # Verify each tree: import resolves inside it, the hot file actually runs,
    # and the .pyc gets compiled here rather than during a measured launch.
    for root in (A, B):
        for bench in benches:
            n, p = BENCHES[bench]
            v = worker(root, n, p, 1, f"verify:{bench}", "0", args.pin, verify=True)
            v.update(kind="verify", tree=root, bench=bench)
            emit(v)
            if v.get("error") or not v.get("hot_executed"):
                emit({"kind": "fatal", "stage": "verify", "tree": root, "rec": v})
                return 4

    for q in range(args.quartets):
        c = canary(args.pin)
        emit({"kind": "canary", "q": q, "rec": c})
        seed = SEEDS[q % len(SEEDS)]
        order = ["A", "B", "B", "A"] if q % 2 == 0 else ["B", "A", "A", "B"]
        for bench in benches:
            n, p = BENCHES[bench]
            for pos, arm in enumerate(order):
                rec = worker(A if arm == "A" else B, n, p, args.m, f"{bench}:{arm}", seed, args.pin)
                rec.update(kind="launch", mode=args.mode, bench=bench, arm=arm, q=q, pos=pos)
                emit(rec)
        print(f"[{args.mode}] quartet {q + 1}/{args.quartets} done, canary "
              f"{c.get('min_all', float('nan')):.4f}s", flush=True)

    emit({"kind": "done", "finished_at": time.time()})
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
