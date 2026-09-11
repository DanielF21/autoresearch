"""Profile the referee target in a sailbox and bring every dump home.

Usage: uv run python scripts/profile_target.py
Writes runs/profile/<ts>/.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import sail

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

INBOX = pathlib.Path(__file__).parent.parent / "inbox"
SHA = "c94928ed9489"
URL = "https://github.com/networkx/networkx"


def main() -> int:
    require("SAIL_API_KEY")
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    image = sail.Image.debian_amd64.apt_install("git", "curl").pip_install("numpy", "scipy", "pandas")
    app = sail.App.find("autoresearch", mint_if_missing=True)
    name = f"profile-{ts}"
    outdir = pathlib.Path("runs/profile") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"creating {name}...", flush=True)
    sb = sail.Sailbox.create(app=app, name=name, image=image, size="m",
                             disk_limit_gib=32, timeout=3600)
    try:
        sb.fs.write("/workspace/harness/profile_target.py",
                    (INBOX / "profile_target.py").read_bytes())
        r = sb.run(f"git clone -q --depth 50 {URL} /workspace/nx_a && "
                   f"cd /workspace/nx_a && git checkout -q {SHA}", timeout=600)
        if r.exit_code:
            raise RuntimeError(f"clone failed: {(r.stderr or '')[-800:]}")
        sb.run("pip uninstall -y networkx", timeout=120)  # never shadow the tree

        r = sb.run("cd /workspace/harness && python3 profile_target.py "
                   "--root /workspace/nx_a --runs 3 --out /workspace/profile", timeout=1800)
        print((r.stdout or "")[-1500:], flush=True)
        if r.exit_code:
            raise RuntimeError(f"profiler failed: {(r.stderr or '')[-2000:]}")

        files = sb.run("ls /workspace/profile", timeout=60).stdout.split()
        for f in files:
            (outdir / f).write_bytes(sb.fs.read(f"/workspace/profile/{f}"))
        print(f"fetched {len(files)} files to {outdir}", flush=True)
    finally:
        sb.terminate()
        print(f"{name} terminated", flush=True)

    s = json.loads((outdir / "summary.json").read_text())
    for bench, b in s["benches"].items():
        print(f"\n=== {bench}: n={b['n']} p={b['p']} edges={b['edges']} ===")
        print(f"  unprofiled {b['plain_min_s']*1e3:.0f}ms, profiled "
              f"{[round(x*1e3) for x in b['profiled_wall_s']]}ms -> cProfile overhead "
              f"{b['overhead_x']:.2f}x")
        print(f"  {b['n_functions']} functions, .prof {b['prof_bytes']:,} bytes")
        print("  self time by kind: " + ", ".join(
            f"{k} {100*v:.1f}%" for k, v in sorted(b['tottime_share_by_kind'].items(), key=lambda kv: -kv[1])))
        print(f"  {'function':62s} {'kind':9s} {'calls':>9s} {'self%':>6s} {'range':>13s} {'ceil self':>9s} {'ceil+callees':>12s}")
        for r in b["rows"][:14]:
            cs = f"{r['ceiling_self']:.2f}x" if r["ceiling_self"] else "n/a"
            cc = f"{r['ceiling_with_callees']:.2f}x" if r["ceiling_with_callees"] else "n/a"
            print(f"  {r['function'][:62]:62s} {r['kind']:9s} {r['calls']:>9,} "
                  f"{100*r['tottime_share']:5.1f}% "
                  f"{100*r['tottime_share_min']:5.1f}-{100*r['tottime_share_max']:4.1f}% "
                  f"{cs:>9s} {cc:>12s}")
        print("  text renderings (chars, lines):")
        for v, t in b["texts"].items():
            print(f"    {v:16s} {t['chars']:>8,} chars {t['lines']:>6,} lines "
                  f"(with absolute paths {t['chars_rawpaths']:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
