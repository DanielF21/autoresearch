"""The run loop: initialise or resume a run directory, then rounds until done.

A run directory is a git repository of its own. After every round it is
committed, so the volume holds a history of the run and ``fetch`` can pull it.

Resume is implicit: the loop starts after the last round in rounds.jsonl. An
attempt directory left behind by a crashed round is detected and the run
refuses to continue until it is moved aside, because a half written round
cannot be trusted.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path

from autoresearch import history
from autoresearch.boxes.protocol import BoxFactory
from autoresearch.config import RunConfig
from autoresearch.orchestrator.pool import RefereePool
from autoresearch.orchestrator.round import run_round
from autoresearch.worker.protocol import Worker

RUN_GITIGNORE = "LOCK\n"


class RunError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _log(log: Path | None, line: str) -> None:
    """One line to the run log, flushed. Nothing else reports progress live."""
    if log is None:
        return
    with log.open("a") as fh:
        fh.write(line + "\n")


def _git(cwd: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False, env=env)


def init_run(
    config: RunConfig, run_dir: Path, docs: tuple[tuple[str, str], ...]
) -> history.RunPaths:
    """Create the run directory. Idempotent: an existing run is left as it is."""
    paths = history.RunPaths(run_dir)
    if paths.config.exists():
        existing = paths.config.read_text()
        if existing != config.source_text:
            raise RunError(
                f"{run_dir} was created with a different config; a run's config never changes"
            )
        return paths
    run_dir.mkdir(parents=True, exist_ok=True)
    paths.config.write_text(config.source_text)
    paths.target.mkdir(exist_ok=True)
    for name, text in docs:
        (paths.target / name).write_text(text)
    paths.attempts.mkdir(exist_ok=True)
    (run_dir / ".gitignore").write_text(RUN_GITIGNORE)
    _git(run_dir, "init", "-q")
    _git(run_dir, "config", "user.email", "orchestrator@autoresearch")
    _git(run_dir, "config", "user.name", "autoresearch")
    commit_run(paths, "run initialised")
    return paths


def commit_run(paths: history.RunPaths, message: str) -> None:
    _git(paths.root, "add", "-A")
    _git(paths.root, "commit", "-q", "-m", message)


def check_consistent(paths: history.RunPaths, config: RunConfig) -> int:
    """The round to start from, or RunError if the directory is half written."""
    last = history.last_completed_round(paths)
    expected = last * config.width
    on_disk = len(history.load_history(paths))
    if on_disk != expected:
        raise RunError(
            f"{on_disk} attempts on disk but {last} completed rounds of width {config.width} "
            f"account for {expected}; move the extra attempt directories aside before resuming"
        )
    return last + 1


def run(
    config: RunConfig,
    paths: history.RunPaths,
    worker: Worker,
    boxes: BoxFactory,
    *,
    until_round: int | None = None,
    log: Path | None = None,
) -> int:
    """Run rounds from the next incomplete one to ``until_round`` or the config's total.

    Returns the last completed round. Referee boxes stay alive between rounds
    and are terminated when the configured number of rounds is reached.
    """
    history.acquire_lock(paths)
    try:
        start = check_consistent(paths, config)
        stop = min(config.rounds, until_round or config.rounds)
        if start > stop:
            return start - 1
        pool = RefereePool(config, paths, boxes)
        pool.start()
        last = start - 1
        for round_no in range(start, stop + 1):
            # A start line, not only a finish line. The log used to gain nothing
            # until a round completed, so a round that never completed was
            # indistinguishable from one that had not started, and a stalled run
            # looked exactly like a slow one. Timestamped, so how long a round
            # has been going is readable without attaching to the process.
            _log(log, f"round {round_no}: workers started at {_now()}")
            outcome = run_round(round_no, config, paths, worker, pool)
            last = round_no
            commit_run(paths, f"round {round_no}")
            r = outcome.record
            best = "none" if r.best_ratio_so_far is None else f"{r.best_ratio_so_far:.4f}"
            slower = sorted(n for n, m in outcome.measurements.items() if m.regressions)
            _log(
                log,
                f"round {r.round}: attempts {list(r.attempt_numbers)} measured "
                f"{list(r.measured_numbers)} real speedups {list(r.clears_noise_numbers)} "
                f"slower somewhere {slower} "
                f"best so far {best} worker {r.worker_wall_s:.0f}s referee "
                f"{r.referee_wall_s:.0f}s errors {len(r.errors)} done at {_now()}",
            )
        if last >= config.rounds:
            pool.terminate_all()
            commit_run(paths, "run complete")
        return last
    finally:
        history.release_lock(paths)
