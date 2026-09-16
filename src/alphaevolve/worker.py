"""One AlphaEvolve candidate as a harness worker attempt.

The harness's round loop calls ``attempt`` once per slot, on threads, and does
the rest: it writes the attempt, has the referee measure the patch, and records
the round. An attempt here is:

1. Rebuild the database from every attempt before this batch (cached, and
   extended as batches finish), and copy it, so a slot's sampling touches
   nothing another slot reads.
2. Sample a parent and inspirations for the slot's island, slot ``w`` on island
   ``w % num_islands``, as OpenEvolve pins iterations to islands, with a random
   generator seeded by the run's seed and the attempt number.
3. Slot 0 only: one meta prompt call that writes a new instruction.
4. One candidate call with no tools. Its SEARCH/REPLACE edits are applied to the
   parent's blocks and the files become the patch.

Every call's tokens are in the attempt's usage, including a reply that yields no
program. A failed call's tokens are unknown, as in the harness's agent loop.

Tracing is the harness's (``autoresearch.observe``): one Voyage per attempt, with
a ``sample`` event naming the parent and the programs shown, a ``model.turn`` per
call, a ``result`` event, and the end. A tracer can never end an attempt.
"""

from __future__ import annotations

import copy
import json
import random
import threading
import time
from typing import Any

from alphaevolve import blocks as blk
from alphaevolve import meta, prompt, records
from alphaevolve.base import EvolveState
from alphaevolve.config import EvolveRunConfig
from alphaevolve.database import Candidate, Database, Program, extend, rebuild
from alphaevolve.diff import build_patch
from autoresearch import history, observe
from autoresearch.model.protocol import ChatModel, ModelError
from autoresearch.observe import AttemptTrace, NullTracer, Tracer
from autoresearch.types import Attempt, StopReason, Usage, WorkerOutput
from autoresearch.worker.protocol import WorkerInput

CLIP = 4000  # as the agent loop clips its transcript
PREVIOUS_ATTEMPTS = 3  # OpenEvolve's evolution history shows the last three


def _clip(s: str, n: int = CLIP) -> str:
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more]"


def _record(**fields: Any) -> str:
    fields["t"] = time.time()
    return json.dumps(fields, sort_keys=True)


class EvolveWorker:
    def __init__(
        self,
        model: ChatModel,
        config: EvolveRunConfig,
        paths: history.RunPaths,
        state: EvolveState,
        tracer: Tracer | None = None,
    ) -> None:
        self._model = model
        self._config = config
        self._paths = paths
        self._state = state
        self._tracer: Tracer = tracer if tracer is not None else NullTracer()
        self._lock = threading.Lock()
        self._numbers: tuple[int, ...] = ()
        self._db: Database | None = None
        self._lineages: dict[int, records.Lineage] = {}

    def _candidate(self, a: Attempt) -> Candidate:
        lineage = self._lineages[a.ref.number]
        texts = lineage.texts if a.patch is not None else None
        return Candidate(a.ref.number, a.ref.worker, texts, records.fitness(a))

    def _snapshot(self, past: tuple[Attempt, ...]) -> tuple[Database, dict[int, records.Lineage]]:
        numbers = tuple(a.ref.number for a in past)
        with self._lock:
            if self._db is None or numbers[: len(self._numbers)] != self._numbers:
                self._db = rebuild(self._config.evolve, self._state.base_texts, ())
                self._numbers = ()
                self._lineages = {}
            fresh = past[len(self._numbers) :]
            for a in fresh:
                self._lineages[a.ref.number] = records.read_lineage(self._paths, a.ref)
            extend(self._db, [self._candidate(a) for a in fresh])
            self._numbers = numbers
            return copy.deepcopy(self._db), dict(self._lineages)

    def attempt(self, inp: WorkerInput) -> WorkerOutput:
        trace = observe.start_attempt(self._tracer, inp.ref, self._config.run.run_id, inp.base_sha)
        try:
            out = self._run(inp, trace)
        except Exception as e:
            # A Voyage left open would look like an attempt still running.
            trace.end(StopReason.MODEL_ERROR, error=f"the worker raised {e!r}")
            raise
        trace.end(out.stop_reason, patch=out.patch, error="" if out.patch else out.error)
        return out

    def _run(self, inp: WorkerInput, trace: AttemptTrace) -> WorkerOutput:
        t0 = time.perf_counter()
        ev = self._config.evolve
        state = self._state
        db, lineages = self._snapshot(inp.history)
        attempts = {a.ref.number: a for a in inp.history}
        db.rng = random.Random(f"{ev.seed}:{inp.ref.number}")

        island = inp.ref.worker % ev.num_islands
        parent, sampled = db.sample_from_island(island, ev.num_diverse_programs)
        ranked = db.get_top_programs(len(db.islands[parent.island]), parent.island)
        shown = ranked[: ev.num_top_programs + ev.num_diverse_programs]
        top = shown[: ev.num_top_programs]
        rest = shown[ev.num_top_programs :]
        diverse = db.rng.sample(rest, min(ev.num_diverse_programs, len(rest))) if rest else []
        seen = {p.id for p in top} | {p.id for p in diverse}
        inspirations = [p for p in sampled if p.id not in seen]
        previous = list(reversed(top[-PREVIOUS_ATTEMPTS:]))

        fitness = {n: records.fitness(a) for n, a in attempts.items()}
        pool = meta.population(lineages, fitness)
        chosen = meta.choose(pool, db.rng)
        trace.tool(
            0,
            "sample",
            {
                "island": island,
                "parent": parent.id,
                "top": [p.id for p in top],
                "diverse": [p.id for p in diverse],
                "inspirations": [p.id for p in inspirations],
                "meta": chosen.id,
                "programs_in_database": len(db.programs),
            },
            f"parent {parent.id} at fitness {parent.fitness:.4f}",
        )

        def view(p: Program) -> prompt.Shown:
            return prompt.Shown(p.id, p.fitness, p.texts, attempts.get(p.number))

        usage = Usage()
        calls = 0
        lines: list[str] = []
        notes: list[str] = []
        generated = ""
        if inp.ref.worker == 0:
            try:
                meta_request = prompt.meta_messages(state.system, state.prefix, pool)
                resp = self._model.complete(meta_request, [], cache_key=inp.cache_key)
                usage = usage + resp.usage
                calls += 1
                trace.model_turn(calls, meta_request, resp)
                generated = meta.parse_generated(resp.content)
                lines.append(
                    _record(
                        kind="meta",
                        usage=resp.usage.to_dict(),
                        finish=resp.finish_reason,
                        content=_clip(resp.content),
                    )
                )
                if not generated:
                    notes.append("the meta prompt reply held no instructions block")
            except ModelError as e:
                notes.append(f"meta prompt call failed: {e}")

        variable = prompt.variable_part(
            chosen.text,
            view(parent),
            [view(p) for p in previous],
            [view(p) for p in top],
            [view(p) for p in diverse],
            [view(p) for p in inspirations],
            state.blocks,
        )
        lines.append(_record(kind="request", variable=_clip(variable)))

        def finish(
            stop: StopReason,
            *,
            patch: str | None = None,
            texts: tuple[str, ...] | None = None,
            rationale: str = "",
            error: str = "",
        ) -> WorkerOutput:
            message = "; ".join(x for x in (*notes, error) if x)
            outcome = f"patch of {len(patch)} bytes" if patch else "no program"
            trace.tool(
                calls, "result", {"stop": str(stop)}, "; ".join(x for x in (outcome, message) if x)
            )
            lineage = records.Lineage(
                island=island,
                parent=parent.id,
                parent_number=parent.number,
                top=tuple(p.id for p in top),
                diverse=tuple(p.id for p in diverse),
                inspirations=tuple(p.id for p in inspirations),
                meta=chosen.id,
                meta_generated=generated,
                texts=texts,
                error=message,
            )
            return WorkerOutput(
                patch=patch,
                prediction=None,
                rationale=rationale,
                stop_reason=stop,
                turns=calls,
                usage=usage,
                wall_s=time.perf_counter() - t0,
                error=message,
                transcript="\n".join([lineage.to_line(), *lines]) + "\n",
            )

        request = prompt.candidate_messages(state.system, state.prefix, variable)
        try:
            resp = self._model.complete(request, [], cache_key=inp.cache_key)
        except ModelError as e:
            return finish(StopReason.MODEL_ERROR, error=str(e))
        usage = usage + resp.usage
        calls += 1
        trace.model_turn(calls, request, resp)
        lines.append(
            _record(
                kind="model",
                usage=resp.usage.to_dict(),
                finish=resp.finish_reason,
                latency_s=round(resp.latency_s, 3),
                reasoning=_clip(resp.reasoning),
                content=_clip(resp.content),
            )
        )
        rationale = _clip(blk.strip_edits(resp.content))
        if resp.finish_reason == "length":
            return finish(
                StopReason.NO_PROGRESS,
                rationale=rationale,
                error="the reply was cut off at the length limit",
            )
        try:
            texts = blk.apply_edits(parent.texts, blk.parse_edits(resp.content))
        except blk.EditError as e:
            return finish(StopReason.NO_PROGRESS, rationale=rationale, error=str(e))
        patch = build_patch(state.sources, blk.assemble(state.sources, state.blocks, texts))
        if patch is None:
            return finish(
                StopReason.NO_PROGRESS,
                rationale=rationale,
                error="the edits leave the code unchanged",
            )
        return finish(StopReason.SUBMITTED, patch=patch, texts=texts, rationale=rationale)
