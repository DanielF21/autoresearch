"""One launch of the target against one source tree. Prints one JSON object.

The tree is imported by path, never installed, so two trees can be timed in
turn from the same box. The launch refuses to run if the package resolves
outside the requested tree.

Everything about the target comes from the flags; nothing is defaulted here.
``--setup`` is Python statements run once in a namespace holding the package
under ``--alias`` and ``ROOT``, a ``pathlib.Path`` of the tree. ``--call`` is
one expression evaluated in that namespace, and that is what is timed.

The result is fingerprinted so both trees can be shown to compute the same
thing. The fingerprint is a hash of a canonical rendering: mappings in key
order, floats rounded, sets sorted, array like objects through ``tolist``. An
object whose only rendering is an address cannot be compared across launches
and is an error, as is a result that is an iterator, since timing a call that
returns a generator times nothing. ``--fingerprint`` is an expression over
``result`` for a target whose result needs its own reduction.

Modes:

- run (default): ``--repeats`` timed calls, each wrapped in the provenance
  counters so a sample that saw steal, throttling, a page fault, a CPU
  migration or memory pressure is marked contaminated.
- ``--verify``: profile one call and report whether the hot file executed and
  what share of self time it took. Also warms the ``.pyc`` so no later launch
  pays compilation.

Both modes report ``fixed_s``, the age of this process when the first timed
call began: interpreter start, the import and the setup. It is read from
``/proc`` and is None where there is none.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from collections.abc import Mapping, Sequence, Set
from typing import Any

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

FLOAT_PLACES = 9


class UnfingerprintableError(ValueError):
    """A result with no rendering that two launches could agree on."""


def canonical(value: Any) -> str:
    """A rendering of ``value`` that depends on its content and nothing else."""
    if value is None or isinstance(value, bool | int | str | bytes):
        return repr(value)
    if isinstance(value, float):
        return repr(round(value, FLOAT_PLACES))
    if isinstance(value, Mapping):
        items = sorted((canonical(k), canonical(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, Set):
        return "{" + ",".join(sorted(canonical(v) for v in value)) + "}"
    if isinstance(value, Sequence):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    if hasattr(value, "__next__"):
        raise UnfingerprintableError(
            "the result is an iterator, so the timed call did no work; "
            "wrap the call in list() or otherwise consume it"
        )
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return canonical(to_list())
    text = repr(value)
    if " at 0x" in text:
        raise UnfingerprintableError(
            f"the result renders as an address ({text[:60]}), which no two launches "
            "share; give the target a 'fingerprint' expression that reduces it"
        )
    return text


def fingerprint(value: Any) -> str:
    """A short hash of the canonical rendering. Raises UnfingerprintableError."""
    return hashlib.md5(canonical(value).encode()).hexdigest()[:10]


def process_age_s() -> float | None:
    """Seconds since this interpreter started, from /proc. None without /proc."""
    try:
        stat = pathlib.Path("/proc/self/stat").read_text()
        uptime = float(pathlib.Path("/proc/uptime").read_text().split()[0])
    except OSError:
        return None
    # The command name (field 2) may contain spaces; everything after its
    # closing parenthesis is fixed width. Field 22, starttime, is then index 19.
    fields = stat.rsplit(")", 1)[1].split()
    start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
    return uptime - start


def _load_provenance() -> Any:
    """The sibling provenance module, found through this file's own directory."""
    import provenance  # type: ignore[import-not-found]

    return provenance


def _pyc_fresh(hot: pathlib.Path) -> bool:
    pyc = pathlib.Path(importlib.util.cache_from_source(str(hot)))
    return pyc.exists() and pyc.stat().st_mtime >= hot.stat().st_mtime


def _fail(**record: Any) -> int:
    print(json.dumps(record))
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="source tree to import from")
    ap.add_argument("--package", required=True, help="import name of the package under test")
    ap.add_argument(
        "--alias", required=True, help="name the package is bound to for setup and call"
    )
    ap.add_argument(
        "--package-root", required=True, help="directory under root put on sys.path, '.' for flat"
    )
    ap.add_argument(
        "--setup", required=True, help="statements run once, with the alias and ROOT bound"
    )
    ap.add_argument("--call", required=True, help="expression to time, in the setup namespace")
    ap.add_argument("--hot", required=True, help="the hot file, relative to root")
    ap.add_argument("--fingerprint", default="", help="expression over result, else canonical")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--label", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-counters", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    hot = (root / args.hot).resolve()
    pyc_fresh = _pyc_fresh(hot) if hot.exists() else False

    sys.path.insert(0, str((root / args.package_root).resolve()))
    t0 = time.perf_counter()
    module = importlib.import_module(args.package)
    import_s = time.perf_counter() - t0
    package_file = pathlib.Path(str(module.__file__)).resolve()
    if root not in package_file.parents:
        return _fail(
            error="package resolved outside the requested tree",
            package_file=str(package_file),
            root=str(root),
        )

    scope: dict[str, Any] = {args.alias: module, "ROOT": root}
    t0 = time.perf_counter()
    exec(args.setup, scope)
    setup_s = time.perf_counter() - t0

    base: dict[str, Any] = {
        "label": args.label,
        "root": str(root),
        "package_file": str(package_file),
        "pyc_fresh_before_import": pyc_fresh,
        "pid": os.getpid(),
        "hash_seed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "python": list(sys.version_info[:3]),
        "import_s": import_s,
        "setup_s": setup_s,
    }

    def fp_of(result: Any) -> str:
        if hasattr(result, "__next__"):
            raise UnfingerprintableError(
                "the call returned an iterator, so nothing was computed inside the "
                "timed region; wrap the call in list() or otherwise consume it"
            )
        reduced = eval(args.fingerprint, dict(scope, result=result)) if args.fingerprint else result
        return fingerprint(reduced)

    if args.verify:
        import cProfile
        import pstats

        base["fixed_s"] = process_age_s()
        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        result = eval(args.call, scope)
        profiler.disable()
        call_s = time.perf_counter() - t0
        stats = pstats.Stats(profiler)
        files: set[pathlib.Path] = set()
        hot_tottime = 0.0
        total_tt = 0.0
        for (fname, _line, _fn), (_cc, _nc, tt, _ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
            total_tt += tt
            if fname and not fname.startswith(("<", "~")):
                try:
                    rp = pathlib.Path(fname).resolve()
                except OSError:
                    continue
                files.add(rp)
                if rp == hot:
                    hot_tottime += tt
        try:
            result_fp = fp_of(result)
        except UnfingerprintableError as e:
            return _fail(error=str(e), **base)
        base.update(
            kind="verify",
            hot_executed=hot in files,
            hot_tottime_share=hot_tottime / total_tt if total_tt else 0.0,
            call_s=call_s,
            result_fp=result_fp,
        )
        print(json.dumps(base))
        return 0

    prov = None if args.no_counters else _load_provenance()

    gc.freeze()
    samples: list[dict[str, Any]] = []
    result = None
    base["fixed_s"] = process_age_s()
    for _ in range(args.repeats):
        gc.collect()
        before = prov.counters() if prov is not None else None
        t0 = time.perf_counter()
        result = eval(args.call, scope)
        t1 = time.perf_counter()
        if prov is not None and before is not None:
            d = prov.delta(before, prov.counters())
            bad, reasons = prov.is_contaminated(d)
            samples.append({"t": t1 - t0, "contaminated": bad, "reasons": reasons})
        else:
            samples.append({"t": t1 - t0, "contaminated": False, "reasons": []})

    try:
        result_fp = fp_of(result)
    except UnfingerprintableError as e:
        return _fail(error=str(e), **base)
    clean: list[float] = [float(s["t"]) for s in samples if not s["contaminated"]]
    base.update(
        kind="run",
        repeats=args.repeats,
        samples=samples,
        min_clean=min(clean) if clean else None,
        min_all=min(float(s["t"]) for s in samples),
        n_contaminated=len(samples) - len(clean),
        result_fp=result_fp,
    )
    print(json.dumps(base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
