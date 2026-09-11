"""networkx intake. Builds it in a box, enumerates benchmarks, times them,
then checkpoints the box so C and D can fork from a ready state.

Usage: uv run python scripts/intake.py
"""
from __future__ import annotations
import datetime as dt, json, pathlib, sys
import sail
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

INBOX = pathlib.Path(__file__).parent.parent / "inbox"
FILES = ["provenance.py", "workload.py", "bench_worker.py", "intake_networkx.py"]
REMOTE = "/workspace/harness"


def parse(text):
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    require("SAIL_API_KEY")
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    image = (sail.Image.debian_amd64
             .apt_install("git", "curl", "build-essential", "valgrind", "time", "stress-ng")
             .pip_install("numpy", "scipy", "pytest"))
    app = sail.App.find("autoresearch", mint_if_missing=True)
    print(f"creating intake-{ts}...", flush=True)
    sb = sail.Sailbox.create(app=app, name=f"intake-{ts}", image=image, size="m",
                             disk_limit_gib=64, timeout=10800)
    recs = []
    try:
        for f in FILES:
            sb.fs.write(f"{REMOTE}/{f}", (INBOX / f).read_bytes())
        base = f"cd {REMOTE} && python3 intake_networkx.py"

        print("building networkx (clone + install)...", flush=True)
        r = sb.run(f"{base} --mode build --skip-tests", timeout=1200)
        recs += parse(r.stdout)
        b = next((x for x in recs if x.get("kind") == "build"), {})
        print(f"  HEAD={b.get('head_sha','?')[:12]}")
        for s in b.get("steps", []):
            print(f"  {s['step']:9s} rc={s['rc']} {s['s']:7.1f}s")
            if s["rc"] != 0:
                print(f"      {s['tail'][-400:]}")

        # Checkpoint the moment networkx is importable, BEFORE anything that can
        # stall. C and D fork from here; losing it to a later timeout is the one
        # unrecoverable failure in this script.
        ck = sb.checkpoint(name=f"networkx-ready-{ts}")
        recs.append({"kind": "checkpoint", "checkpoint_id": ck.checkpoint_id})
        print(f"  checkpoint saved: {ck.checkpoint_id}", flush=True)

        print("\nrunning test suite (capped 12 min, parallel)...", flush=True)
        r = sb.run(f"cd /workspace/networkx && pip install pytest-xdist -q && "
                   f"timeout 700 python -m pytest networkx -q --no-header "
                   f"-p no:cacheprovider -n 4 2>&1 | tail -6", timeout=900)
        print("  " + (r.stdout or "").strip()[-500:], flush=True)
        recs.append({"kind": "pytest", "out": (r.stdout or "")[-1500:]})

        print("\nenumerating benchmarks...", flush=True)
        r = sb.run(f"{base} --mode enumerate", timeout=900)
        recs += parse(r.stdout)
        tot = 0
        for m in [x for x in recs if x.get("kind") == "module"]:
            for bm in m["benchmarks"]:
                n = len(bm["methods"]) * bm["n_param_combos"]
                tot += n
                print(f"  {m['module']}.{bm['class']}: {len(bm['methods'])} methods "
                      f"x {bm['n_param_combos']} params = {n}")
        print(f"  total benchmark instances: {tot}")

        print("\ntiming benchmarks (hard caps: 3s each, 150s per module)...", flush=True)
        sb.run(f"{base} --mode time_all --out /workspace/timings.jsonl "
               f"--cap 3 --module-timeout 150", timeout=1500)
    except Exception as e:  # noqa: BLE001
        print(f"!! step failed: {e!r}", file=sys.stderr)
    finally:
        # Always fetch whatever was written. Nothing depends on the run finishing.
        try:
            recs += parse(sb.fs.read("/workspace/timings.jsonl").decode())
        except Exception as e:  # noqa: BLE001
            print(f"  (no timings file: {e!r})", file=sys.stderr)
        sb.terminate()
        print("intake box terminated")

    outdir = pathlib.Path("runs/intake"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{ts}.jsonl").write_text("\n".join(json.dumps(x) for x in recs))

    T = [x for x in recs if x.get("kind") == "timing"]
    errs = [x for x in recs if x.get("kind") == "timing_error"]
    band = sorted([x for x in T if x["in_band"]], key=lambda x: -x["min_s"])
    print(f"\n=== {len(T)} timed, {len(errs)} errored, {len(band)} in the 50ms-2s band ===")
    print(f"{'benchmark':62s} {'time':>9s} {'setup':>9s}")
    for x in band[:25]:
        label = f"{x['class']}.{x['method']}({','.join(x['combo'])})"
        print(f"  {label[:60]:60s} {x['min_s']*1e3:8.1f}ms {x['setup_s']*1e3:8.1f}ms")
    print(f"\nwrote {outdir}/{ts}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
