"""One candidate attempt against a scripted model, and whole runs through the harness loop."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from alphaevolve import base, budget, records
from alphaevolve import config as config_mod
from alphaevolve.config import EvolveRunConfig
from alphaevolve.worker import EvolveWorker
from autoresearch import history
from autoresearch.boxes.protocol import BoxError
from autoresearch.model.fake_model import FakeChatModel, text
from autoresearch.model.protocol import Message, ModelError, ModelResponse
from autoresearch.orchestrator import run as run_mod
from autoresearch.orchestrator import status as status_mod
from autoresearch.patch import changed_files
from autoresearch.types import AttemptRef, StopReason, Usage
from autoresearch.worker.protocol import WorkerInput
from tests.ae_helpers import (
    FLAT,
    HOT,
    SOURCE,
    config_text,
    edit_reply,
    evolve_factory,
    flat_name,
    setup_run,
)

EDIT = edit_reply("        total += 1", "        total += 2")


def _input(cfg: EvolveRunConfig, slot: int, number: int = 1) -> WorkerInput:
    return WorkerInput(
        ref=AttemptRef(number, 1, slot),
        base_sha=cfg.run.target.sha,
        target=cfg.run.target,
        history=(),
        docs=(),
        cache_key="key",
    )


def _one(tmp_path: Path, slot: int, script: list[object]) -> tuple[object, FakeChatModel]:
    cfg, paths, _, state = setup_run(tmp_path)
    model = FakeChatModel(script=script)  # type: ignore[arg-type]
    out = EvolveWorker(model, cfg, paths, state).attempt(_input(cfg, slot))
    return out, model


def test_a_candidate_is_one_call_with_no_tools_and_its_edits_become_the_patch(
    tmp_path: Path,
) -> None:
    cfg, paths, _, state = setup_run(tmp_path)
    model = FakeChatModel(script=[text(EDIT)])
    out = EvolveWorker(model, cfg, paths, state).attempt(_input(cfg, 1))
    assert out.stop_reason == StopReason.SUBMITTED and out.turns == 1
    assert out.patch is not None and changed_files(out.patch) == (HOT,)
    assert "+        total += 2" in out.patch
    assert out.usage == Usage(100, 0, 10, 0) and out.rationale == "Faster.\n[edit]"
    assert model.offered == [[]] and model.cache_keys == ["key"]
    system, user = model.requests[0]
    assert system["content"] == state.system and user["content"].startswith(state.prefix)
    lineage = records.Lineage.from_dict(json.loads(out.transcript.splitlines()[0]))
    assert lineage.parent == "base" and lineage.island == 1 and lineage.meta == "m0000"
    assert lineage.texts is not None and "total += 2" in lineage.texts[0]


def test_slot_zero_writes_a_meta_prompt_first_and_pays_for_it(tmp_path: Path) -> None:
    out, model = _one(
        tmp_path, 0, [text("<instructions>Hoist the loop.</instructions>"), text(EDIT)]
    )
    assert out.stop_reason == StopReason.SUBMITTED and out.turns == 2  # type: ignore[attr-defined]
    assert out.usage == Usage(200, 0, 20, 0)  # type: ignore[attr-defined]
    assert "# Task: write instructions" in model.requests[0][1]["content"]
    lineage = records.Lineage.from_dict(json.loads(out.transcript.splitlines()[0]))  # type: ignore[attr-defined]
    assert lineage.meta_generated == "Hoist the loop."


def test_a_failed_meta_call_leaves_the_candidate_to_run(tmp_path: Path) -> None:
    out, _ = _one(tmp_path, 0, [ModelError("meta down"), text(EDIT)])
    assert out.stop_reason == StopReason.SUBMITTED  # type: ignore[attr-defined]
    assert out.usage == Usage(100, 0, 10, 0) and "meta prompt call failed" in out.error  # type: ignore[attr-defined]


def test_a_failed_candidate_call_is_a_model_error(tmp_path: Path) -> None:
    out, _ = _one(tmp_path, 1, [ModelError("endpoint down")])
    assert out.stop_reason == StopReason.MODEL_ERROR and out.patch is None  # type: ignore[attr-defined]
    assert out.usage == Usage() and "endpoint down" in out.error  # type: ignore[attr-defined]


def test_replies_that_make_no_program_keep_their_tokens(tmp_path: Path) -> None:
    cut: ModelResponse = replace(text(EDIT), finish_reason="length")
    cases = [
        (cut, "cut off"),
        (text("I would change the loop."), "no SEARCH/REPLACE"),
        (text(edit_reply("        total += 1", "        total += 1")), "unchanged"),
        (text(edit_reply("    nowhere()", "x")), "matches no marked block"),
    ]
    for i, (reply, why) in enumerate(cases):
        out, _ = _one(tmp_path / str(i), 1, [reply])
        assert out.stop_reason == StopReason.NO_PROGRESS and out.patch is None  # type: ignore[attr-defined]
        assert out.usage == Usage(100, 0, 10, 0) and why in out.error  # type: ignore[attr-defined]
        first = json.loads(out.transcript.splitlines()[0])  # type: ignore[attr-defined]
        assert first["kind"] == "lineage" and first["texts"] is None


def _responder() -> object:
    counter = itertools.count(1)

    def respond(messages: list[Message]) -> ModelResponse:
        if "# Task: write instructions" in messages[-1]["content"]:
            return text("<instructions>Hoist the loop.</instructions>")
        n = next(counter)
        return text(edit_reply("def clustering(G):", f"def clustering(G):\n    pass  # FAST {n}"))

    return respond


def _model() -> FakeChatModel:
    return FakeChatModel(script=[_responder()] * 200)  # type: ignore[list-item]


def test_a_run_stops_after_the_batch_that_spends_the_budget(tmp_path: Path) -> None:
    # Width 2: two candidate calls and one meta call a batch, 110 tokens each.
    cfg, paths, factory, state = setup_run(tmp_path, budget=700)
    model = _model()
    worker = EvolveWorker(model, cfg, paths, state)
    stop = budget.stop_at(state.budget)
    assert run_mod.run(cfg.run, paths, worker, factory, stop=stop) == 3
    assert [r.round for r in history.read_rounds(paths)] == [1, 2, 3]
    assert budget.primary(budget.run_usage(paths)) == 990
    assert all(offered == [] for offered in model.offered)
    assert len(set(model.cache_keys)) == 1
    attempts = history.load_history(paths)
    assert len(attempts) == 6
    metas = {"m0000", "m0001", "m0003"}
    for a in attempts:
        lineage = records.read_lineage(paths, a.ref)
        expected = Usage(200, 0, 20, 0) if a.ref.worker == 0 else Usage(100, 0, 10, 0)
        assert a.usage == expected
        assert a.stop_reason == StopReason.SUBMITTED and a.patch and "FAST" in a.patch
        assert a.measurement is not None and a.measurement.clears_noise
        assert lineage.texts is not None and lineage.meta in metas
    assert all(box.terminated for box in factory.created)
    st = status_mod.compute_status(paths, cfg.run.rounds, cfg.run.worker.model, cfg.run.run_id)
    assert st.attempts == 6 and st.best_ratio == pytest.approx(1.5)
    log = subprocess.run(
        ["git", "log", "--format=%s"], cwd=paths.root, capture_output=True, text=True, check=True
    ).stdout
    assert "run complete" in log

    # A later launch has nothing to spend and creates nothing.
    created = len(factory.created)
    assert (
        run_mod.run(cfg.run, paths, EvolveWorker(model, cfg, paths, state), factory, stop=stop) == 3
    )
    assert len(factory.created) == created and len(history.read_rounds(paths)) == 3


def test_a_run_resumes_from_its_records(tmp_path: Path) -> None:
    cfg, paths, factory, state = setup_run(tmp_path, budget=700)
    stop = budget.stop_at(state.budget)
    assert (
        run_mod.run(
            cfg.run,
            paths,
            EvolveWorker(_model(), cfg, paths, state),
            factory,
            until_round=1,
            stop=stop,
        )
        == 1
    )
    again = base.load(paths)
    assert again == state
    assert (
        run_mod.run(cfg.run, paths, EvolveWorker(_model(), cfg, paths, again), factory, stop=stop)
        == 3
    )
    assert len(history.load_history(paths)) == 6


def test_a_half_written_batch_refuses_to_resume(tmp_path: Path) -> None:
    cfg, paths, factory, state = setup_run(tmp_path, budget=700)
    stop = budget.stop_at(state.budget)
    run_mod.run(
        cfg.run, paths, EvolveWorker(_model(), cfg, paths, state), factory, until_round=1, stop=stop
    )
    shutil.copytree(paths.attempts / "0001", paths.attempts / "0100")
    with pytest.raises(run_mod.RunError, match="move the extra attempt"):
        run_mod.run(cfg.run, paths, EvolveWorker(_model(), cfg, paths, state), factory, stop=stop)


def test_prepare_reads_the_base_once_and_checks_it(tmp_path: Path) -> None:
    cfg, paths, factory, state = setup_run(tmp_path)
    assert [b.function for b in state.blocks] == ["clustering"]
    assert state.sources == {HOT: SOURCE} and state.budget == 700
    assert len(factory.created) == 1 and factory.created[0].terminated
    assert base.prepare(paths, cfg, factory, tmp_path / "runs") == state
    assert len(factory.created) == 1  # read back, no second box

    parsed = config_mod.parse(config_text(700))
    run_cfg = replace(parsed.run, target=replace(parsed.run.target, docs=()))
    other = run_mod.init_run(run_cfg, tmp_path / "bad", ((flat_name(run_cfg), FLAT),))
    lying = evolve_factory({HOT: SOURCE}, blob=lambda _data: "0" * 40)
    with pytest.raises(BoxError, match="read back as blob"):
        base.prepare(other, replace(parsed, run=run_cfg), lying, tmp_path / "runs")
    assert lying.created[0].terminated


def test_prepare_refuses_before_any_box_when_the_profile_or_budget_is_missing(
    tmp_path: Path,
) -> None:
    parsed = config_mod.parse(config_text(700))
    run_cfg = replace(parsed.run, target=replace(parsed.run.target, docs=()))
    factory = evolve_factory({HOT: SOURCE})
    no_docs = run_mod.init_run(run_cfg, tmp_path / "nodocs", ())
    with pytest.raises(base.PrepareError, match="autoresearch profile"):
        base.prepare(no_docs, replace(parsed, run=run_cfg), factory, tmp_path / "runs")
    from_run = replace(parsed.evolve, budget_tokens=None, budget_from_run="harness_x")
    with_docs = run_mod.init_run(run_cfg, tmp_path / "docs", ((flat_name(run_cfg), FLAT),))
    with pytest.raises(budget.BudgetError, match="no completed rounds"):
        base.prepare(
            with_docs, replace(parsed, run=run_cfg, evolve=from_run), factory, tmp_path / "runs"
        )
    assert factory.created == []


def test_a_budget_from_a_harness_run_is_its_prompt_plus_completion(tmp_path: Path) -> None:
    from tests.test_ae_curve import write_round

    paths = history.RunPaths(tmp_path / "runs" / "harness_x")
    paths.attempts.mkdir(parents=True)
    write_round(paths, 1, [1], Usage(1000, 900, 50, 10), [1.2])
    write_round(paths, 2, [2], Usage(2000, 1500, 100, 0), [1.0])
    parsed = config_mod.parse(config_text(700))
    from_run = replace(parsed.evolve, budget_tokens=None, budget_from_run="harness_x")
    assert base.resolve_budget(from_run, tmp_path / "runs") == 3150


def test_an_attempt_is_traced_from_sampling_to_its_end(tmp_path: Path) -> None:
    from autoresearch.model.fake_model import Scripted
    from tests.helpers import FakeTracer

    cfg, paths, _, state = setup_run(tmp_path)
    tracer = FakeTracer()
    script: list[Scripted] = [text("<instructions>Hoist.</instructions>"), text(EDIT)]
    out = EvolveWorker(FakeChatModel(script=script), cfg, paths, state, tracer).attempt(
        _input(cfg, 0)
    )
    assert out.stop_reason == StopReason.SUBMITTED
    trace = tracer.only
    assert trace.kinds == ["tool", "model_turn", "model_turn", "tool", "end"]
    assert trace.calls[0] == ("tool", (0, "sample"))
    assert trace.calls[3] == ("tool", (2, "result")) and trace.calls[4] == ("end", "submitted")


def test_a_broken_tracer_cannot_end_an_attempt(tmp_path: Path) -> None:
    from tests.helpers import FakeTracer

    cfg, paths, _, state = setup_run(tmp_path)
    worker = EvolveWorker(
        FakeChatModel(script=[text(EDIT)]), cfg, paths, state, FakeTracer(raises=True)
    )
    assert worker.attempt(_input(cfg, 1)).stop_reason == StopReason.SUBMITTED
