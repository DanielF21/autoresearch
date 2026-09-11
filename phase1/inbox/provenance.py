"""Provenance: everything needed to decide, in hindsight, whether two numbers
are comparable.

Runs inside the sailbox. Static facts are collected once per box. Dynamic
counters are sampled immediately before and after every timed region, so a run
carries the evidence of its own contamination rather than relying on the box
having been quiet.

Design rule: collect raw, never aggregate here. Aggregation destroys the
ability to re-bucket later, which is the whole point.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import sysconfig
import time

SCHEMA_VERSION = 1


def _read(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _cmd(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _cpuinfo_first(key: str) -> str:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.split(":")[0].strip() == key:
            return line.split(":", 1)[1].strip()
    return ""


# --------------------------------------------------------------------------
# static facts: one snapshot per box
# --------------------------------------------------------------------------

def cpu_facts() -> dict:
    flags = _cpuinfo_first("flags").split()
    # Hash the sorted flag set. This, not the model name, is what decides
    # whether numpy dispatches to the same SIMD kernels and therefore whether
    # two Ir counts are comparable.
    flag_hash = hashlib.md5(" ".join(sorted(flags)).encode()).hexdigest()[:12]

    caches = []
    for idx in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        caches.append({
            "level": _read(f"{idx}/level"),
            "type": _read(f"{idx}/type"),
            "size": _read(f"{idx}/size"),
            "shared_cpu_list": _read(f"{idx}/shared_cpu_list"),
        })

    numa = {
        os.path.basename(n): _read(f"{n}/cpulist")
        for n in sorted(glob.glob("/sys/devices/system/node/node*"))
        if os.path.isdir(n)
    }

    return {
        "model_name": _cpuinfo_first("model name"),
        "family": _cpuinfo_first("cpu family"),
        "model": _cpuinfo_first("model"),
        "stepping": _cpuinfo_first("stepping"),
        "microcode": _cpuinfo_first("microcode"),
        "mhz_nominal": _cpuinfo_first("cpu MHz"),
        "flags": sorted(flags),          # full list, so a diff shows WHICH flag moved
        "flag_hash": flag_hash,
        "caches": caches,                # L3 size moves wall clock for memory-bound graph work
        "numa_nodes": numa,
        "nproc": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "smt_active": _read("/sys/devices/system/cpu/smt/active", "unknown"),
        "thread_siblings": sorted({
            _read(p) for p in glob.glob("/sys/devices/system/cpu/cpu*/topology/thread_siblings_list")
        }),
    }


def mitigation_facts() -> dict:
    """Spectre/MDS mitigation state.

    CPython's eval loop is exceptionally indirect-branch heavy (computed goto
    dispatch), so retpoline vs IBRS vs off is easily a 10%+ swing on exactly
    this workload. As load-bearing as the SIMD flags and almost never recorded.
    """
    vulns = {
        os.path.basename(p): _read(p)
        for p in sorted(glob.glob("/sys/devices/system/cpu/vulnerabilities/*"))
    }
    return {
        "vulnerabilities": vulns,
        "vuln_hash": hashlib.md5(json.dumps(vulns, sort_keys=True).encode()).hexdigest()[:12],
        "kernel_cmdline": _read("/proc/cmdline"),
    }


def python_facts() -> dict:
    """A PGO/LTO CPython differs 10-20% from a plain build, and its instruction
    counts differ too. If the image ever rebuilds onto a different interpreter,
    every number shifts and this is the only thing that would show it.
    """
    keys = ("CONFIG_ARGS", "PY_CFLAGS", "PY_CFLAGS_NODIST", "WITH_COMPUTED_GOTOS",
            "OPT", "CC", "LTOFLAGS", "PGO_PROF_USE_FLAG", "Py_DEBUG", "Py_ENABLE_SHARED")
    cfg = {k: str(sysconfig.get_config_var(k)) for k in keys}
    return {
        "version": sys.version,
        "version_info": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable,
        "config_vars": cfg,
        "config_hash": hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
        "gc_enabled_at_import": True,
        "hash_seed_env": os.environ.get("PYTHONHASHSEED", "<unset>"),
    }


def platform_facts() -> dict:
    # cgroup v2 first, then v1. This host is v1; do not assume either.
    quota = _read("/sys/fs/cgroup/cpu.max")
    if not quota:
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        quota = f"v1 {q} {p}"
        cgroup_version = "v1"
    else:
        cgroup_version = "v2"
    return {
        "kernel": platform.release(),
        "uname": " ".join(platform.uname()),
        "boot_id": _read("/proc/sys/kernel/random/boot_id"),
        "machine_id": _read("/etc/machine-id"),
        "cgroup_version": cgroup_version,
        "cpu_quota": quota,
        "clocksource": _read("/sys/devices/system/clocksource/clocksource0/current_clocksource"),
        "aslr": _read("/proc/sys/kernel/randomize_va_space"),
        "perf_event_paranoid": _read("/proc/sys/kernel/perf_event_paranoid"),
        "valgrind": _cmd(["valgrind", "--version"]),
        "sailbox_name": os.environ.get("SAILBOX_NAME", "<unset>"),
    }


def package_facts() -> dict:
    freeze = _cmd([sys.executable, "-m", "pip", "freeze"])
    return {
        "pip_freeze": freeze.splitlines(),
        "pip_freeze_hash": hashlib.md5(freeze.encode()).hexdigest()[:12],
    }


def static_facts() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at": time.time(),
        "cpu": cpu_facts(),
        "mitigations": mitigation_facts(),
        "python": python_facts(),
        "platform": platform_facts(),
        "packages": package_facts(),
    }


# --------------------------------------------------------------------------
# dynamic counters: sampled around every timed region
# --------------------------------------------------------------------------

_libc = None
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
except Exception:  # noqa: BLE001
    _libc = None


def sched_getcpu() -> int:
    """Which CPU the process is actually on. Confirms a taskset pin held."""
    if _libc is not None and hasattr(_libc, "sched_getcpu"):
        try:
            return int(_libc.sched_getcpu())
        except Exception:  # noqa: BLE001
            pass
    try:  # field 39 of /proc/self/stat is the last-run processor
        return int(_read("/proc/self/stat").split()[38])
    except Exception:  # noqa: BLE001
        return -1


def _proc_status_ctxt() -> tuple[int, int]:
    vol = nonvol = -1
    for line in _read("/proc/self/status").splitlines():
        if line.startswith("voluntary_ctxt_switches"):
            vol = int(line.split()[-1])
        elif line.startswith("nonvoluntary_ctxt_switches"):
            nonvol = int(line.split()[-1])
    return vol, nonvol


def counters() -> dict:
    """Cheap enough to call around every timed region."""
    steal = -1
    for line in _read("/proc/stat").splitlines():
        if line.startswith("cpu "):
            parts = line.split()
            if len(parts) > 8:
                steal = int(parts[8])       # field 9 of the line; parts[0] is "cpu"
            break

    throttled = {}
    stat = _read("/sys/fs/cgroup/cpu.stat") or _read("/sys/fs/cgroup/cpu/cpu.stat")
    for line in stat.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("nr_periods", "nr_throttled", "throttled_time",
                                            "throttled_usec"):
            throttled[parts[0]] = int(parts[1])

    psi = {}
    for line in _read("/proc/pressure/cpu").splitlines():
        kind = line.split()[0] if line.split() else ""
        for tok in line.split()[1:]:
            if "=" in tok:
                k, _, v = tok.partition("=")
                psi[f"{kind}_{k}"] = float(v)

    vol, nonvol = _proc_status_ctxt()
    ru = resource.getrusage(resource.RUSAGE_SELF)

    return {
        "t_wall": time.time(),
        "t_mono": time.monotonic(),
        "steal": steal,
        "cgroup": throttled,
        "psi": psi,
        "cpu": sched_getcpu(),
        "ctxt_voluntary": vol,
        "ctxt_nonvoluntary": nonvol,      # preemption during a timed region
        "ru_minflt": ru.ru_minflt,
        "ru_majflt": ru.ru_majflt,
        "ru_nvcsw": ru.ru_nvcsw,
        "ru_nivcsw": ru.ru_nivcsw,
        "loadavg": os.getloadavg(),
    }


def delta(before: dict, after: dict) -> dict:
    """What changed across the timed region. Non-zero on the wrong field means
    the run is contaminated and should be excluded, not averaged in."""
    d = {
        "steal": after["steal"] - before["steal"],
        "ctxt_nonvoluntary": after["ctxt_nonvoluntary"] - before["ctxt_nonvoluntary"],
        "ctxt_voluntary": after["ctxt_voluntary"] - before["ctxt_voluntary"],
        "ru_minflt": after["ru_minflt"] - before["ru_minflt"],
        "ru_majflt": after["ru_majflt"] - before["ru_majflt"],
        "ru_nivcsw": after["ru_nivcsw"] - before["ru_nivcsw"],
        "elapsed": after["t_mono"] - before["t_mono"],
        "cpu_before": before["cpu"],
        "cpu_after": after["cpu"],
        "cpu_migrated": before["cpu"] != after["cpu"],
        "psi_some_avg10_after": after["psi"].get("some_avg10"),
    }
    for k in set(before["cgroup"]) | set(after["cgroup"]):
        d[f"cgroup_{k}"] = after["cgroup"].get(k, 0) - before["cgroup"].get(k, 0)
    return d


def is_contaminated(d: dict) -> tuple[bool, list[str]]:
    """Fixed in advance. A run that trips any of these is excluded and the
    exclusion rate is reported."""
    reasons = []
    if d.get("steal", 0) > 0:
        reasons.append("steal")
    if d.get("cgroup_nr_throttled", 0) > 0:
        reasons.append("throttled")
    if d.get("ru_majflt", 0) > 0:
        reasons.append("major_page_fault")
    if d.get("cpu_migrated"):
        reasons.append("cpu_migration")
    # delta() stores this key as "psi_some_avg10_after". The original check read
    # "psi_some_avg10", which never exists, so the PSI guard never fired in A or B.
    if (d.get("psi_some_avg10_after") or 0) > 1.0:
        reasons.append("psi")
    return (len(reasons) > 0), reasons


if __name__ == "__main__":
    json.dump(static_facts(), sys.stdout, indent=2)
    print()
