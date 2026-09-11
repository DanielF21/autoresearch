"""networkx intake: clone, build, enumerate benchmarks, time them.

Runs inside the box. Produces the facts needed to choose a referee target:
which benchmarks land in a usable duration band, and how expensive setup is.

ASV is deliberately bypassed. `asv run` builds its own per-commit environments
and applies its own repeat autotuning, neither of which the referee can hold
fixed across arms. The benchmark BODIES are what matter, so we import the
classes directly and drive them ourselves.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import itertools
import json
import os
import pathlib
import resource
import signal
import subprocess
import sys
import time

REPO = pathlib.Path("/workspace/networkx")
BENCH_DIR = REPO / "benchmarks" / "benchmarks"


def sh(cmd: str, timeout: int = 3600) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)[-4000:]


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def enumerate_benchmarks(mod) -> list[dict]:
    """ASV convention: class has `params` and `param_names`; setup(self, *params)
    and time_*(self, *params) take the same parameter tuple."""
    found = []
    for cname in dir(mod):
        cls = getattr(mod, cname)
        if not isinstance(cls, type) or cname.startswith("_"):
            continue
        methods = [m for m in dir(cls) if m.startswith("time_")]
        if not methods:
            continue
        params = getattr(cls, "params", [])
        if params and not isinstance(params[0], (list, tuple)):
            params = [params]
        combos = list(itertools.product(*params)) if params else [()]
        found.append({"class": cname, "methods": methods, "n_param_combos": len(combos),
                      "param_names": getattr(cls, "param_names", []),
                      "combos": [list(map(str, c)) for c in combos]})
    return found


class Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise Timeout()


def time_benchmark(mod, cname: str, method: str, combo, reps: int = 1,
                   cap_s: int = 3) -> dict:
    cls = getattr(mod, cname)
    inst = cls()
    signal.signal(signal.SIGALRM, _alarm)
    t_setup0 = time.perf_counter()
    signal.alarm(cap_s * 4)          # setup gets a looser cap than the body
    try:
        if hasattr(inst, "setup"):
            inst.setup(*combo)
    finally:
        signal.alarm(0)
    t_setup = time.perf_counter() - t_setup0

    fn = getattr(inst, method)
    times = []
    # One rep first. An expensive benchmark should be discovered, not run 3 times.
    for i in range(reps):
        gc.collect()
        signal.alarm(cap_s)
        t0 = time.perf_counter()
        try:
            fn(*combo)
        finally:
            signal.alarm(0)
        times.append(time.perf_counter() - t0)
        if i == 0 and times[0] > cap_s / 3:
            break
    return {"class": cname, "method": method, "combo": list(map(str, combo)),
            "setup_s": t_setup, "min_s": min(times), "times": times, "reps_run": len(times)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["build", "enumerate", "time", "time_all"], required=True)
    ap.add_argument("--module", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=int, default=3, help="per-benchmark seconds")
    ap.add_argument("--module-timeout", type=int, default=150)
    ap.add_argument("--mem-gib", type=int, default=4)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--ref", default="main")
    ap.add_argument("--lo", type=float, default=0.05, help="duration band low, seconds")
    ap.add_argument("--hi", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    sink = open(args.out, "a") if args.out else None

    def emit(o):
        line = json.dumps(o)
        if sink:
            sink.write(line + "\n")
            sink.flush()      # nothing may depend on this script finishing
        else:
            print(line, flush=True)

    if args.mode == "time_all":
        # One subprocess per module. Benchmark graphs are built at import time and
        # some are enormous, so a module that stalls or exhausts memory must not
        # take the run down with it.
        for path in sorted(BENCH_DIR.glob("*.py")):
            if path.stem.startswith("_"):
                continue
            t0 = time.time()
            cmd = [sys.executable, __file__, "--mode", "time", "--module", path.stem,
                   "--out", args.out, "--cap", str(args.cap), "--mem-gib", str(args.mem_gib)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=args.module_timeout)
                emit({"kind": "module_done", "module": path.stem, "rc": r.returncode,
                      "s": time.time() - t0, "tail": (r.stdout + r.stderr)[-500:]})
            except subprocess.TimeoutExpired:
                emit({"kind": "module_timeout", "module": path.stem,
                      "s": time.time() - t0, "cap": args.module_timeout})
        return 0

    if args.mode == "build":
        steps = []
        t0 = time.time()
        rc, out = sh(f"git clone --depth 50 https://github.com/networkx/networkx {REPO}")
        steps.append({"step": "clone", "rc": rc, "s": time.time() - t0, "tail": out[-400:]})

        t0 = time.time()
        rc, sha = sh(f"cd {REPO} && git rev-parse HEAD")
        head = sha.strip()

        t0 = time.time()
        rc, out = sh(f"cd {REPO} && pip install -e . numpy scipy 2>&1 | tail -5", timeout=1800)
        steps.append({"step": "install", "rc": rc, "s": time.time() - t0, "tail": out[-600:]})

        t0 = time.time()
        rc, out = sh(f"cd {REPO} && python -c 'import networkx; print(networkx.__version__)'")
        steps.append({"step": "import", "rc": rc, "s": time.time() - t0, "tail": out[-200:]})

        emit({"kind": "build", "head_sha": head, "steps": steps})
        return 0

    if args.mem_gib:
        cap = args.mem_gib * (1 << 30)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError):
            pass

    # benchmark_algorithms.py does `import benchmarks...`, so the parent of the
    # package directory must be on the path too, not just the package directory.
    sys.path.insert(0, str(BENCH_DIR))
    sys.path.insert(0, str(BENCH_DIR.parent))
    sys.path.insert(0, str(REPO / "benchmarks"))
    paths = [p for p in sorted(BENCH_DIR.glob("*.py")) if not p.stem.startswith("_")]
    if args.module:
        paths = [p for p in paths if p.stem == args.module]
    mods = {}
    for p_ in paths:
        t0 = time.time()
        try:
            mods[p_.stem] = load_module(p_)
            emit({"kind": "module_loaded", "module": p_.stem, "s": time.time() - t0})
        except (MemoryError, Exception) as e:  # noqa: BLE001
            emit({"kind": "module_load_failed", "module": p_.stem,
                  "s": time.time() - t0, "error": repr(e)[:300]})

    if args.mode == "enumerate":
        for name, mod in mods.items():
            emit({"kind": "module", "module": name, "benchmarks": enumerate_benchmarks(mod)})
        return 0

    # mode == time
    for name, mod in mods.items():
        for b in enumerate_benchmarks(mod):
            for method in b["methods"]:
                for combo_s in b["combos"]:
                    cls = getattr(mod, b["class"])
                    raw = getattr(cls, "params", [])
                    if raw and not isinstance(raw[0], (list, tuple)):
                        raw = [raw]
                    combo = next((c for c in itertools.product(*raw)
                                  if list(map(str, c)) == combo_s), ()) if raw else ()
                    try:
                        rec = time_benchmark(mod, b["class"], method, combo,
                                             args.reps, args.cap)
                        rec["module"] = name
                        rec["in_band"] = args.lo <= rec["min_s"] <= args.hi
                        emit({"kind": "timing", **rec})
                    except Timeout:
                        emit({"kind": "timing_skipped", "module": name, "class": b["class"],
                              "method": method, "combo": combo_s, "cap_s": args.cap,
                              "reason": "exceeded per-benchmark cap"})
                    except MemoryError:
                        emit({"kind": "timing_skipped", "module": name, "class": b["class"],
                              "method": method, "combo": combo_s, "reason": "out of memory"})
                    except Exception as e:  # noqa: BLE001
                        emit({"kind": "timing_error", "module": name, "class": b["class"],
                              "method": method, "combo": combo_s, "error": repr(e)[:300]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
