"""The run loop: initialise or resume a run directory, then rounds until done.

A run directory is a git repository of its own. After every round it is
committed, so the volume holds a history of the run and ``fetch`` can pull it.
The nested incumbent repository is ignored by that outer repository and is
rebuilt from the accepted patches if it is ever missing.

Resume is implicit: the loop starts after the last round in rounds.jsonl. An
attempt directory left behind by a crashed round is detected and the run
refuses to continue until it is moved aside, because a half written round
cannot be trusted.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from autoresearch import history, incumbent
from autoresearch.boxes.protocol import BoxFactory
from autoresearch.config import RunConfig
from autoresearch.orchestrator.pool import RefereePool
from autoresearch.orchestrator.round import rebuild_incumbent, run_round
from autoresearch.worker.protocol import Worker

RUN_GITIGNORE = "incumbent/\nLOCK\n"


class RunError(RuntimeError):
    pass


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
        rebuild_incumbent(config, paths)
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
    rebuild_incumbent(config, paths)
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
        inc = paths.incumbent
        pool.sync_all(
            config.target.sha,
            incumbent.cumulative_diff(inc, config.target.sha),
            incumbent.tree_hash(inc),
        )
        last = start - 1
        for round_no in range(start, stop + 1):
            outcome = run_round(round_no, config, paths, worker, pool)
            last = round_no
            commit_run(paths, f"round {round_no}")
            if log is not None:
                with log.open("a") as fh:
                    r = outcome.record
                    fh.write(
                        f"round {r.round}: attempts {list(r.attempt_numbers)} accepted "
                        f"{list(r.accepted_numbers)} worker {r.worker_wall_s:.0f}s referee "
                        f"{r.referee_wall_s:.0f}s errors {len(r.errors)}\n"
                    )
        if last >= config.rounds:
            pool.terminate_all()
            commit_run(paths, "run complete")
        return last
    finally:
        history.release_lock(paths)
