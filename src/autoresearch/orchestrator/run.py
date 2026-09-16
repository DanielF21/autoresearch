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
import sys
from collections.abc import Callable
from pathlib import Path

from autoresearch import history
from autoresearch.boxes.protocol import BoxError, BoxFactory
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


def _log_teardown(log: Path | None, failures: tuple[str, ...]) -> None:
    """Every referee that may still be running, to the run log and to stderr."""
    if not failures:
        _log(log, f"referee boxes terminated at {_now()}")
        return
    for line in failures:
        message = f"referee box may still be running: {line}"
        _log(log, message)
        print(message, file=sys.stderr, flush=True)


def run(
    config: RunConfig,
    paths: history.RunPaths,
    worker: Worker,
    boxes: BoxFactory,
    *,
    until_round: int | None = None,
    log: Path | None = None,
    stop: Callable[[history.RunPaths], bool] | None = None,
) -> int:
    """Run rounds from the next incomplete one to ``until_round`` or the config's total.

    Returns the last completed round. Every referee box is terminated before this
    returns or raises, whatever ended it: the last round, ``until_round``, an
    exception, or a signal ``autoresearch.shutdown`` turned into one. Nothing is
    kept for a later launch, which builds its own referees. A box that will not
    terminate is logged, left in boxes.json, and raised as a BoxError, except
    when an exception is already on its way out, which is never replaced.

    ``stop`` ends the run early: it is read before the first round and after
    every round, and a run it ends is complete. None, the default, is every run
    the harness itself launches; the AlphaEvolve comparison passes a token budget.
    """
    history.acquire_lock(paths)
    pool = RefereePool(config, paths, boxes)
    try:
        try:
            last, stopped = _rounds(config, paths, worker, pool, until_round, log, stop)
        except BaseException:
            _log_teardown(log, pool.terminate_all())
            raise
        failures = pool.terminate_all()
        _log_teardown(log, failures)
        if last >= config.rounds or stopped:
            commit_run(paths, "run complete")
        if failures:
            raise BoxError(
                f"{len(failures)} referee boxes may still be running, recorded in "
                f"{paths.boxes}: {'; '.join(failures)}"
            )
        return last
    finally:
        history.release_lock(paths)


def _rounds(
    config: RunConfig,
    paths: history.RunPaths,
    worker: Worker,
    pool: RefereePool,
    until_round: int | None,
    log: Path | None,
    stop: Callable[[history.RunPaths], bool] | None = None,
) -> tuple[int, bool]:
    """The rounds themselves, and whether ``stop`` ended them. Teardown is the
    caller's, so no exit path here can skip it."""
    start = check_consistent(paths, config)
    final = min(config.rounds, until_round or config.rounds)
    if start > final:
        return start - 1, False
    if stop is not None and stop(paths):
        _log(log, f"stop condition already met before round {start}; nothing to run")
        return start - 1, True
    pool.start()
    last = start - 1
    for round_no in range(start, final + 1):
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
        if len(r.errors) >= config.width:
            # Every worker failed before the model could work: a box that would
            # not come up, or a model endpoint that refused. That is the platform,
            # not the experiment, and it takes seconds per attempt, so a run left
            # alone would spend its remaining rounds recording nothing. In t1_w4c
            # round 2 all four workers lost their box within twenty seconds and
            # round 3 started at once. The round is on disk and committed, so
            # the run resumes from the next one when the cause is gone.
            message = (
                f"round {round_no}: every one of {config.width} workers failed; stopping "
                f"so the remaining rounds are not spent on a platform fault. "
                f"First error: {r.errors[0][:300]}"
            )
            _log(log, message)
            raise RunError(message)
        if stop is not None and stop(paths):
            _log(log, f"round {round_no}: stop condition met, run complete at {_now()}")
            return last, True
    return last, False
