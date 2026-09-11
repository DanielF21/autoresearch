"""Phase 0: characterize a sailbox before trusting any timing from it.

Three of the four measurement decision rules branch on this output, so it runs
first and its output is recorded verbatim. Writes runs/phase0/<ts>.md.

Usage: uv run python scripts/phase0_characterize.py [--keep]
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import sail

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _env import require  # noqa: E402

IMAGE_PKGS_APT = ("git", "curl", "build-essential", "valgrind", "linux-perf", "time", "stress-ng")
IMAGE_PKGS_PIP = ("pyperf",)

# (label, command). Ordered so the decision-relevant reads come first.
CHECKS: list[tuple[str, str]] = [
    # This host is cgroup v1 (tmpfs at /sys/fs/cgroup, no unified cpu.max). Try v2 first,
    # fall back to v1, because the harness must not assume either.
    ("cgroup version", "stat -fc %T /sys/fs/cgroup; ls /sys/fs/cgroup | tr '\\n' ' '"),
    ("cpu quota", "cat /sys/fs/cgroup/cpu.max 2>/dev/null || { echo -n 'v1 quota/period: '; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us | tr '\\n' ' '; echo; }"),
    ("cpu throttle counters", "cat /sys/fs/cgroup/cpu.stat 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.stat"),
    ("steal (field 9 of cpu line)", "awk '/^cpu /{print $9}' /proc/stat"),
    ("boot settle wait", "for i in $(seq 1 60); do ps aux | grep -q '[e]xt4lazyinit' || break; sleep 2; done; echo \"settled after ${i}x2s\""),
    ("psi cpu (post settle)", "cat /proc/pressure/cpu 2>/dev/null || echo 'PSI MISSING'"),
    ("nproc / nproc --all", "echo \"$(nproc) / $(nproc --all)\""),
    ("vcpu affinity mask", "taskset -cp $$ 2>/dev/null || echo 'taskset MISSING'"),
    ("smt siblings", "cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list 2>/dev/null | sort -u || echo MISSING"),
    ("smt active", "cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo MISSING"),
    ("cpu model", "grep -m1 'model name' /proc/cpuinfo; grep -m1 microcode /proc/cpuinfo"),
    ("tsc flags", "grep -oE 'constant_tsc|nonstop_tsc|tsc_reliable' /proc/cpuinfo | sort -u"),
    ("clocksource", "cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null || echo MISSING"),
    ("virt type", "systemd-detect-virt 2>/dev/null; grep -m1 -o hypervisor /proc/cpuinfo || true"),
    ("dmi identity", "cat /sys/devices/virtual/dmi/id/sys_vendor /sys/devices/virtual/dmi/id/product_name 2>/dev/null || echo MISSING"),
    ("cpufreq control", "ls /sys/devices/system/cpu/cpu0/cpufreq/ 2>&1 | head -5"),
    ("perf_event_paranoid", "cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo MISSING"),
    ("PMU via perf stat", "perf stat -e instructions,cycles true 2>&1 || echo 'perf MISSING'"),
    ("aslr setting", "cat /proc/sys/kernel/randomize_va_space"),
    ("busy processes", "ps aux --sort=-%cpu | head -8"),
    ("loadavg", "cat /proc/loadavg"),
    ("kernel / python", "uname -r; python3 -VV"),
    ("valgrind", "valgrind --version 2>&1 | head -1"),
    ("memory", "free -g | head -2"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="leave the sailbox running")
    ap.add_argument("--size", default="m")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    require("SAIL_API_KEY")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.name or f"phase0-{ts}"

    image = sail.Image.debian_amd64.apt_install(*IMAGE_PKGS_APT).pip_install(*IMAGE_PKGS_PIP)

    print(f"building image and creating sailbox '{name}' (size={args.size})...", flush=True)
    app = sail.App.find("autoresearch", mint_if_missing=True)
    sb = sail.Sailbox.create(
        app=app, name=name, image=image, size=args.size,
        memory_limit_gib=16, disk_limit_gib=64, timeout=3600,
    )
    print(f"  up: vcpu={sb.vcpu_count} mem_mib={sb.memory_mib} arch={sb.architecture}", flush=True)

    lines: list[str] = [
        f"# Phase 0 characterization: {name}",
        "",
        f"- taken: {dt.datetime.now().isoformat()}",
        f"- size: {args.size}, vcpu_count: {sb.vcpu_count}, memory_mib: {sb.memory_mib}",
        f"- architecture: {sb.architecture}",
        "",
    ]

    for label, cmd in CHECKS:
        try:
            r = sb.run(cmd, timeout=120)
            out = (r.stdout or "").rstrip()
            err = (r.stderr or "").rstrip()
            body = out if out else err
            if out and err:
                body = f"{out}\n[stderr] {err}"
        except Exception as exc:  # noqa: BLE001 - record the failure, do not abort the sweep
            body = f"<command raised: {exc!r}>"
        print(f"  [{label}]")
        lines += [f"## {label}", "", "```", f"$ {cmd}", body or "<empty>", "```", ""]

    outdir = pathlib.Path("runs/phase0")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{ts}.md"
    path.write_text("\n".join(lines))
    print(f"\nwrote {path}")

    if args.keep:
        print(f"sailbox '{name}' left running. terminate with Sailbox.get(...).terminate()")
    else:
        sb.terminate()
        print("sailbox terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
