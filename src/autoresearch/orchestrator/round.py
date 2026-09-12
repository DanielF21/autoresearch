"""One round: fan out the workers, judge every patch, stack what was accepted.

The round is the unit of history. Every worker in it sees the same history and
the same incumbent, and none sees another's patch until the round is over.
The orchestrator is the only writer: attempts are written once, results once,
and the incumbent advances by one commit per accepted patch.

Stacking, when more than one patch is accepted in a round: the best median
ratio goes in first. Each remaining accepted patch is re judged against the
new incumbent on its own referee. One that no longer applies or no longer
clears the threshold is recorded as superseded.
"""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from autoresearch import history, incumbent
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig
from autoresearch.orchestrator.pool import RefereePool
from autoresearch.patch import normalised_hash
from autoresearch.types import (
    AttemptRef,
    RefereeResult,
    RoundRecord,
    StopReason,
    Usage,
    Verdict,
    WorkerOutput,
)
from autoresearch.worker.protocol import Worker, WorkerInput


@dataclass(frozen=True)
class RoundOutcome:
    record: RoundRecord
    results: dict[int, RefereeResult]


def load_docs(paths: history.RunPaths) -> tuple[tuple[str, str], ...]:
    if not paths.target.exists():
        return ()
    return tuple((p.name, p.read_text()) for p in sorted(paths.target.iterdir()) if p.is_file())


def _judge_one(
    slot: int, pool: RefereePool, incumbent_sha: str, patch: str
) -> tuple[RefereeResult, str]:
    """Judge on the slot's referee. A box failure rebuilds the box and retries once."""
    for attempt in range(2):
        ref = pool.get(slot)
        try:
            result = ref.judge(incumbent_sha, patch)
            if ref.broken:
                pool.rebuild(slot)
            return result, ""
        except BoxError as e:
            if attempt == 0:
                pool.rebuild(slot)
                continue
            return (
                RefereeResult(Verdict.FAILED, f"box error: {e}", 0.0),
                f"referee {slot} failed twice: {e}",
            )
    raise AssertionError("unreachable")


def run_round(
    round_no: int,
    config: RunConfig,
    paths: history.RunPaths,
    worker: Worker,
    pool: RefereePool,
) -> RoundOutcome:
    t_round = time.perf_counter()
    target = config.target
    inc = paths.incumbent
    sha_before = incumbent.head_sha(inc)
    tree_before = incumbent.tree_hash(inc)
    stack_before = incumbent.cumulative_diff(inc, target.sha)
    past = history.load_history(paths)
    docs = load_docs(paths)
    first_number = history.next_attempt_number(paths)
    seen_hashes = {normalised_hash(a.patch): a.ref.dirname for a in past if a.patch}

    inputs = [
        WorkerInput(
            ref=AttemptRef(number=first_number + w, round=round_no, worker=w),
            incumbent_sha=sha_before,
            incumbent_tree=tree_before,
            stack_diff=stack_before,
            target=target,
            history=past,
            docs=docs,
            cache_key=f"{config.run_id}-{config.config_hash}",
        )
        for w in range(config.width)
    ]

    # 1. Workers, concurrently. Each creates and destroys its own box.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.width) as ex:
        outputs: list[WorkerOutput] = list(ex.map(worker.attempt, inputs))
    worker_wall = time.perf_counter() - t0

    for inp, out in zip(inputs, outputs, strict=True):
        history.write_attempt(
            paths,
            inp.ref,
            sha_before,
            {
                "config_hash": config.config_hash,
                "history_numbers": [a.ref.number for a in past],
                "cache_key": inp.cache_key,
            },
            out,
            out.transcript,
        )

    # 2. Decide what needs a referee: no patch and duplicates are settled here.
    results: dict[int, RefereeResult] = {}
    to_judge: dict[int, str] = {}
    errors: list[str] = []
    for w, out in enumerate(outputs):
        if out.stop_reason in (StopReason.BOX_ERROR, StopReason.MODEL_ERROR):
            errors.append(f"worker {w}: {out.stop_reason}: {out.error[:200]}")
        if not out.patch or not out.patch.strip():
            results[w] = RefereeResult(
                Verdict.NO_PATCH, f"worker stopped with {out.stop_reason}", config.referee.threshold
            )
            continue
        h = normalised_hash(out.patch)
        if h in seen_hashes:
            results[w] = RefereeResult(
                Verdict.DUPLICATE, f"same as attempt {seen_hashes[h]}", config.referee.threshold
            )
            continue
        seen_hashes[h] = inputs[w].ref.dirname
        to_judge[w] = out.patch

    # 3. Referees, concurrently, one per slot.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(to_judge))) as ex:
        futures = {
            w: ex.submit(_judge_one, w, pool, sha_before, patch) for w, patch in to_judge.items()
        }
        for w, fut in futures.items():
            result, err = fut.result()
            results[w] = result
            if err:
                errors.append(err)
    referee_wall = time.perf_counter() - t0

    # 4. Stack accepted patches into the incumbent, best first, re judging the rest.
    accepted = sorted(
        (w for w, r in results.items() if r.verdict == Verdict.ACCEPTED),
        key=lambda w: results[w].median_ratio or 0.0,
        reverse=True,
    )
    merged: list[int] = []
    for w in accepted:
        patch = outputs[w].patch or ""
        if merged:
            if not incumbent.applies_cleanly(inc, patch):
                results[w] = RefereeResult(
                    Verdict.SUPERSEDED,
                    f"accepted at {results[w].median_ratio:.4f} but no longer applies after attempt "
                    f"{inputs[merged[-1]].ref.dirname}",
                    config.referee.threshold,
                    pairs=results[w].pairs,
                    median_ratio=results[w].median_ratio,
                )
                continue
            pool.get(w).sync_incumbent(
                target.sha, incumbent.cumulative_diff(inc, target.sha), incumbent.tree_hash(inc)
            )
            again, err = _judge_one(w, pool, incumbent.head_sha(inc), patch)
            if err:
                errors.append(err)
            if again.verdict != Verdict.ACCEPTED:
                results[w] = RefereeResult(
                    Verdict.SUPERSEDED,
                    f"accepted at {results[w].median_ratio:.4f} alone, {again.reason} on the new incumbent",
                    config.referee.threshold,
                    pairs=again.pairs,
                    median_ratio=again.median_ratio,
                    ir=again.ir,
                    tests=again.tests,
                )
                continue
            results[w] = again
        incumbent.apply_and_commit(
            inc, patch, f"attempt {inputs[w].ref.dirname}: {results[w].reason}"
        )
        merged.append(w)

    for w, inp in enumerate(inputs):
        history.write_result(paths, inp.ref, results[w])

    # 5. Every referee to the new incumbent, ready for the next round.
    sha_after = incumbent.head_sha(inc)
    if merged:
        pool.sync_all(
            target.sha, incumbent.cumulative_diff(inc, target.sha), incumbent.tree_hash(inc)
        )

    usage = Usage()
    for out in outputs:
        usage = usage + out.usage
    record = RoundRecord(
        round=round_no,
        attempt_numbers=tuple(i.ref.number for i in inputs),
        incumbent_sha_before=sha_before,
        incumbent_sha_after=sha_after,
        accepted_numbers=tuple(inputs[w].ref.number for w in merged),
        worker_wall_s=worker_wall,
        referee_wall_s=referee_wall,
        usage=usage,
        errors=tuple(errors),
        finished_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    history.append_round(paths, record)
    _ = time.perf_counter() - t_round
    return RoundOutcome(record=record, results=results)


def rebuild_incumbent(config: RunConfig, paths: history.RunPaths) -> str:
    """Recreate incumbent/ from the pinned commit plus every accepted patch in order.

    Used when a run directory arrives without its incumbent, for example after a
    fetch, since the nested git repo is not part of the run's own history.
    """
    if paths.incumbent.exists():
        return incumbent.head_sha(paths.incumbent)
    incumbent.clone_at(config.target.repo, config.target.sha, paths.incumbent)
    for a in history.load_history(paths):
        if a.verdict == Verdict.ACCEPTED and a.patch:
            incumbent.apply_and_commit(paths.incumbent, a.patch, f"attempt {a.ref.dirname}")
    return incumbent.head_sha(paths.incumbent)


def profile_docs_from_repo(root: Path, config: RunConfig) -> tuple[tuple[str, str], ...]:
    """The document files named in the config, read relative to the package repo root."""
    return tuple((Path(p).name, (root / p).read_text()) for p in config.target.docs)
