"""Do concurrently-created sailboxes land on the same CPU?

Instruction counts are only comparable across boxes if the same code path
executes. CPU model itself does not matter; runtime CPUID dispatch does, and
numpy picks its SIMD kernels that way. This provisions N boxes at once and
compares the things that actually decide comparability.

Usage: uv run python scripts/check_cpu_homogeneity.py [-n 4]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import json
import pathlib
import sys

import sail

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

PROBE = r"""
echo "MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "FAMILY=$(grep -m1 '^cpu family' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "MODELNO=$(grep -m1 '^model\s*:' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "STEPPING=$(grep -m1 stepping /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "MICROCODE=$(grep -m1 microcode /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "MHZ=$(grep -m1 'cpu MHz' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "NPROC=$(nproc)"
echo "SIMD=$(grep -m1 ^flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512f|avx512dq|avx512bw|avx512vl|sse4_2|fma|bmi2)$' | sort | tr '\n' ',')"
echo "FLAGHASH=$(cd /workspace && python3 -c 'import provenance as p; print(p.cpu_facts()["flag_hash"])' 2>&1 | tail -1)"
echo "VULNHASH=$(cd /workspace && python3 -c 'import provenance as p; print(p.mitigation_facts()["vuln_hash"])' 2>&1 | tail -1)"
echo "PYCFGHASH=$(cd /workspace && python3 -c 'import provenance as p; print(p.python_facts()["config_hash"])' 2>&1 | tail -1)"
echo "L3=$(cd /workspace && python3 -c 'import provenance as p; print([c["size"] for c in p.cpu_facts()["caches"] if c["level"]==\"3\"])' 2>&1 | tail -1)"
echo "NPVER=$(python3 -c 'import numpy; print(numpy.__version__)' 2>&1 | tail -1)"
echo "NPBASE=$(python3 -c '
try:
    import numpy._core._multiarray_umath as m      # numpy >= 2.0
except ImportError:
    import numpy.core._multiarray_umath as m       # numpy < 2.0
print(",".join(m.__cpu_baseline__))' 2>&1 | tail -1)"
echo "NPDISPATCH=$(python3 -c '
try:
    import numpy._core._multiarray_umath as m
except ImportError:
    import numpy.core._multiarray_umath as m
f = m.__cpu_features__
print(",".join(sorted(k for k in m.__cpu_dispatch__ if f.get(k))))' 2>&1 | tail -1)"
"""


async def probe(app, image, idx: int, size: str) -> tuple[str, dict[str, str]]:
    name = f"cpuprobe-{dt.datetime.now().strftime('%H%M%S')}-{idx}"
    sb = await sail.Sailbox.create.aio(
        app=app, name=name, image=image, size=size, disk_limit_gib=32, timeout=900
    )
    try:
        # Use the same hashing implementation as every other measurement.
        # A hash computed by a different serialization is NOT comparable, and a
        # spurious mismatch is indistinguishable from real hardware drift.
        prov = (pathlib.Path(__file__).parent.parent / "inbox" / "provenance.py").read_bytes()
        await sb.fs.write.aio("/workspace/provenance.py", prov)
        r = await sb.run.aio(PROBE, timeout=180)
        out = {}
        for line in (r.stdout or "").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return name, out
    finally:
        # Do not swallow this. A silent cleanup failure bills until autosleep.
        # Note: Sailbox.list() returns terminated boxes too, so a name appearing
        # in a listing is not evidence the box is alive. Check .status.
        await sb.terminate.aio()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4, help="boxes to provision concurrently")
    ap.add_argument("--size", default="m")
    args = ap.parse_args()
    require("SAIL_API_KEY")

    image = sail.Image.debian_amd64.pip_install("numpy")
    app = await sail.App.find.aio("autoresearch", mint_if_missing=True)

    print(f"provisioning {args.n} boxes (size={args.size}) concurrently...", flush=True)
    results = await asyncio.gather(
        *(probe(app, image, i, args.size) for i in range(args.n)), return_exceptions=True
    )

    rows = [r for r in results if not isinstance(r, BaseException)]
    for r in results:
        if isinstance(r, BaseException):
            print(f"  !! probe failed: {r!r}")

    # The fields that actually decide whether Ir counts are comparable.
    DECIDES = ["MODEL", "FAMILY", "MODELNO", "STEPPING", "FLAGHASH", "VULNHASH",
               "PYCFGHASH", "L3", "SIMD", "NPDISPATCH", "NPBASE"]
    print()
    for name, d in rows:
        print(f"{name}: nproc={d.get('NPROC')} mhz={d.get('MHZ')} model={d.get('MODEL')!r}")
        print(f"    family/model/stepping={d.get('FAMILY')}/{d.get('MODELNO')}/{d.get('STEPPING')} "
              f"microcode={d.get('MICROCODE')}")
        print(f"    flaghash={d.get('FLAGHASH')} simd={d.get('SIMD')}")
        print(f"    numpy baseline={d.get('NPBASE')}")
        print(f"    numpy dispatch={d.get('NPDISPATCH')}")

    print("\n--- homogeneity verdict ---")
    verdict_ok = True
    for field in DECIDES:
        vals = collections.Counter(d.get(field) for _, d in rows)
        uniform = len(vals) == 1
        if field in ("FLAGHASH", "VULNHASH", "PYCFGHASH", "L3", "SIMD",
                     "NPDISPATCH", "NPBASE") and not uniform:
            verdict_ok = False
        print(f"  {field:12s} {'SAME' if uniform else 'DIFFERS'}  {dict(vals)}")

    print()
    if verdict_ok:
        print("PASS: CPU feature set and numpy dispatch identical across boxes.")
        print("      Ir counts are comparable across boxes. Cross-box fan-out is safe.")
    else:
        print("FAIL: feature set or numpy dispatch differs across boxes.")
        print("      Ir counts are NOT comparable as-is. Pin NPY_DISABLE_CPU_FEATURES to a")
        print("      common baseline, or bucket counts by FLAGHASH and only compare within a bucket.")

    outdir = pathlib.Path("runs/cpu-homogeneity")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({n: d for n, d in rows}, indent=2))
    print(f"\nwrote {path}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
