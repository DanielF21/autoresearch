"""Run pytest on one tree and print one JSON summary. The full log goes to ``--log``.

The package directory, ``root / package_root``, is put first on ``PYTHONPATH``
so the tests import the code under test and not some other copy. pytest runs
from ``root`` so the target path is repo relative whatever the layout. Exit
code 0 means every selected test passed.

With more than one worker the tests of one file stay on one worker, in file
order (xdist's ``loadfile``). xdist's default scatters a file's tests across
workers, and a test that leans on a state an earlier test in its file left
behind then passes or fails by the draw: pyparsing's
``Test09_WithLeftRecursionParsing::testIndentedBlockClass2`` passed one night's
check and failed the next morning's twice, on the same commit, while the same
file passed whole on one worker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SUMMARY_RE = re.compile(
    r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)"
)


def parse_summary(text: str) -> dict[str, int]:
    """Counts from pytest's final summary line, such as ``3 passed, 1 failed in 2.1s``."""
    counts: dict[str, int] = {}
    for line in reversed(text.splitlines()):
        if (
            " in " in line
            and SUMMARY_RE.search(line)
            and ("passed" in line or "failed" in line or "error" in line or "no tests ran" in line)
        ):
            for n, key in SUMMARY_RE.findall(line):
                key = {"error": "errors", "warning": "warnings"}.get(key, key)
                counts[key] = int(n)
            break
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--package-root", default=".", help="directory under root the package imports from"
    )
    ap.add_argument("--target", required=True, help="path relative to root: a file or a package")
    ap.add_argument("--scope", required=True, help="label recorded in the summary")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    argv = [
        sys.executable,
        "-m",
        "pytest",
        args.target,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    if args.workers > 1:
        argv += ["-n", str(args.workers), "--dist", "loadfile"]
    env = dict(os.environ, PYTHONPATH=str((root / args.package_root).resolve()))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
            check=False,
        )
        rc = proc.returncode
        text = proc.stdout + "\n" + proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc = -1
        text = (
            (e.stdout or b"").decode(errors="replace")
            + "\n"
            + (e.stderr or b"").decode(errors="replace")
        )
        timed_out = True
    duration = time.perf_counter() - t0
    Path(args.log).write_text(text)
    counts = parse_summary(text)
    out = {
        "kind": "tests",
        "scope": args.scope,
        "rc": rc,
        "timed_out": timed_out,
        "duration_s": duration,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "errors": counts.get("errors", 0),
        "skipped": counts.get("skipped", 0),
        "ok": rc == 0 and not timed_out,
        "tail": text[-1500:],
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
