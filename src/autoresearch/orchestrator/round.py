"""One round: fan out the workers, measure every patch, append everything.

The round is the unit of history. Every worker in it starts from the same base
commit, and none sees another's patch until the round is over. Every worker sees
the same history too, except in the hidden leader experiment, where the top
``[worker].hidden_slots`` slots are not shown the leader set (the best attempt
and its near copies, ``history.leader_set``); what each slot saw is written in
its input record. The orchestrator is the only writer: attempts are written once
and measurements once. Nothing is merged. Every patch, including one that fails
tests or is slower, is measured in full and recorded, because it is context for
the next worker.

The only attempt the referee does not see is one with no patch. A patch whose
normalised diff matches an earlier attempt is recorded as a duplicate of it and
measured anyway, which gives a second sample of the same change.
"""

from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from autoresearch import history
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig
from autoresearch.orchestrator.pool import RefereePool
from autoresearch.patch import normalised_hash
from autoresearch.types import (
    AttemptRef,
    Measurement,
    RoundRecord,
    StopReason,
    Usage,
    WorkerOutput,
)
from autoresearch.worker.protocol import Worker, WorkerInput

NO_PATCH = "no_patch"


@dataclass(frozen=True)
class RoundOutcome:
    record: RoundRecord
    measurements: dict[int, Measurement]


def load_docs(paths: history.RunPaths) -> tuple[tuple[str, str], ...]:
    if not paths.target.exists():
        return ()
    return tuple((p.name, p.read_text()) for p in sorted(paths.target.iterdir()) if p.is_file())


def _measure_one(slot: int, pool: RefereePool, patch: str) -> tuple[Measurement, str]:
    """Measure on the slot's referee. A box failure rebuilds the box and retries once."""
    for attempt in range(2):
        ref = pool.get(slot)
        try:
            measurement = ref.measure(patch)
            if ref.broken:
                pool.rebuild(slot)
            return measurement, ""
        except BoxError as e:
            if attempt == 0:
                pool.rebuild(slot)
                continue
            return (
                Measurement(errors=(f"box error: {e}",)),
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
    target = config.target
    base_sha = target.sha
    past = history.load_history(paths)
    docs = load_docs(paths)
    first_number = history.next_attempt_number(paths)
    seen_hashes = {normalised_hash(a.patch): a.ref.dirname for a in past if a.patch}

    # Slot w runs prompt version prompts[w % len(prompts)]. The cache key carries
    # the version, since each version is its own cached prefix, and whether the
    # slot is hidden, since a hidden slot's first message is its own prefix too.
    versions = config.worker.prompts
    withheld = history.leader_set(past) if config.worker.hidden_slots else frozenset()
    visible = tuple(a for a in past if a.ref.number not in withheld)
    # Tests build configs with replace(), so the width can be below the setting.
    first_hidden = max(0, config.width - config.worker.hidden_slots)
    inputs: list[WorkerInput] = []
    for w in range(config.width):
        hidden = w >= first_hidden
        version = versions[w % len(versions)]
        inputs.append(
            WorkerInput(
                ref=AttemptRef(number=first_number + w, round=round_no, worker=w),
                base_sha=base_sha,
                target=target,
                history=visible if hidden else past,
                docs=docs,
                cache_key=f"{config.run_id}-{config.config_hash}-{version}"
                + ("-hidden" if hidden else ""),
                prompt=version,
                hidden_numbers=tuple(sorted(withheld)) if hidden else (),
            )
        )

    # 1. Workers, concurrently. Each creates and destroys its own box.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.width) as ex:
        outputs: list[WorkerOutput] = list(ex.map(worker.attempt, inputs))
    worker_wall = time.perf_counter() - t0

    # 2. Record every attempt. Only an attempt with no patch skips the referee.
    errors: list[str] = []
    to_measure: dict[int, str] = {}
    for w, (inp, out) in enumerate(zip(inputs, outputs, strict=True)):
        if out.stop_reason in (StopReason.BOX_ERROR, StopReason.MODEL_ERROR):
            errors.append(f"worker {w}: {out.stop_reason}: {out.error[:200]}")
        patch = out.patch if out.patch and out.patch.strip() else None
        duplicate_of = ""
        if patch is not None:
            h = normalised_hash(patch)
            duplicate_of = seen_hashes.get(h, "")
            seen_hashes.setdefault(h, inp.ref.dirname)
            to_measure[w] = patch
        history.write_attempt(
            paths,
            inp.ref,
            base_sha,
            {
                "config_hash": config.config_hash,
                "history_numbers": [a.ref.number for a in inp.history],
                "hidden_numbers": list(inp.hidden_numbers),
                "cache_key": inp.cache_key,
                "prompt": inp.prompt,
            },
            out,
            out.transcript,
            skipped="" if patch is not None else NO_PATCH,
            duplicate_of=duplicate_of,
        )

    # 3. Referees, concurrently, one per slot. Everything that applies is measured in full.
    measurements: dict[int, Measurement] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(to_measure))) as ex:
        futures = {w: ex.submit(_measure_one, w, pool, patch) for w, patch in to_measure.items()}
        for w, fut in futures.items():
            measurement, err = fut.result()
            measurements[w] = measurement
            if err:
                errors.append(err)
    referee_wall = time.perf_counter() - t0

    # 4. Write measurements once, then the round record with best so far.
    for w, m in measurements.items():
        history.write_measurement(paths, inputs[w].ref, m)
    best, _ = history.best_ratio(history.load_history(paths))

    usage = Usage()
    for out in outputs:
        usage = usage + out.usage
    record = RoundRecord(
        round=round_no,
        attempt_numbers=tuple(i.ref.number for i in inputs),
        base_sha=base_sha,
        measured_numbers=tuple(inputs[w].ref.number for w in sorted(measurements)),
        clears_noise_numbers=tuple(
            inputs[w].ref.number for w in sorted(measurements) if measurements[w].clears_noise
        ),
        best_ratio_so_far=best,
        worker_wall_s=worker_wall,
        referee_wall_s=referee_wall,
        usage=usage,
        errors=tuple(errors),
        finished_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )
    history.append_round(paths, record)
    return RoundOutcome(record=record, measurements=measurements)


def profile_docs_from_repo(root: Path, config: RunConfig) -> tuple[tuple[str, str], ...]:
    """The document files named in the config, read relative to the package repo root."""
    return tuple((Path(p).name, (root / p).read_text()) for p in config.target.docs)
