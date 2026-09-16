"""The program database, its rebuild from records, the meta prompt population."""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import replace

from alphaevolve import meta, records
from alphaevolve.config import EvolveConfig
from alphaevolve.database import BASE_FITNESS, BASE_ID, Candidate, extend, rebuild
from alphaevolve.features import fast_code_diversity
from autoresearch.types import (
    Attempt,
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    StopReason,
    SuiteResult,
    Usage,
)
from tests.helpers import BASE_SHA

BASE = ("def f():\n    x = 1\n", "def g():\n    return 0\n")
CFG = EvolveConfig(
    seed=7,
    population_size=1000,
    archive_size=10,
    num_islands=3,
    migration_interval=4,
    feature_bins=10,
    diversity_reference_size=5,
    num_top_programs=3,
    num_diverse_programs=2,
    max_code_length=10_000,
    migration_rate=0.2,
    elite_selection_ratio=0.1,
    exploration_ratio=0.2,
    exploitation_ratio=0.7,
    budget_tokens=1,
    budget_from_run="",
)


def _candidates(n: int, seed: int = 1) -> list[Candidate]:
    rng = random.Random(seed)
    out: list[Candidate] = []
    for i in range(1, n + 1):
        texts = (
            None
            if i % 7 == 0
            else (
                "def f():\n" + "    x = 1\n" * rng.randint(1, 30),
                f"def g():\n    return {rng.random():.3f}\n",
            )
        )
        out.append(Candidate(i, (i - 1) % 6, texts, rng.choice([0.0, 0.0, 1.1, 1.3, 2.0])))
    return out


def test_the_base_is_on_island_zero_at_fitness_one() -> None:
    db = rebuild(CFG, BASE, ())
    assert db.programs[BASE_ID].fitness == BASE_FITNESS and db.programs[BASE_ID].island == 0
    assert db.best_program_id == BASE_ID


def test_the_same_records_rebuild_the_same_database() -> None:
    cands = _candidates(60)
    assert rebuild(CFG, BASE, cands).snapshot() == rebuild(CFG, BASE, cands).snapshot()


def test_extending_round_by_round_equals_rebuilding_once() -> None:
    cands = _candidates(60)
    whole = rebuild(CFG, BASE, cands).snapshot()
    for k in (0, 12, 30, 59):
        db = rebuild(CFG, BASE, cands[:k])
        extend(db, cands[k:])
        assert db.snapshot() == whole


def test_children_go_to_their_slots_island_and_no_program_is_not_added() -> None:
    cands = _candidates(20)
    db = rebuild(replace(CFG, migration_interval=1000), BASE, cands)
    added = [c for c in cands if c.texts is not None]
    assert sum(db.island_generations) == len(added)
    for c in cands:
        pid = f"{c.number:04d}"
        if c.texts is None:
            assert pid not in db.programs
        elif pid in db.programs:
            assert db.programs[pid].island == c.worker % CFG.num_islands


def test_migration_runs_on_the_generation_interval_and_marks_migrants() -> None:
    db = rebuild(CFG, BASE, _candidates(60))
    assert db.last_migration_generation >= CFG.migration_interval
    migrants = [p for p in db.programs.values() if p.migrant]
    assert migrants and all("@m" in p.id for p in migrants)
    quiet = rebuild(replace(CFG, migration_interval=1000), BASE, _candidates(60))
    assert not any(p.migrant for p in quiet.programs.values())


def test_the_population_limit_keeps_the_best() -> None:
    cands = _candidates(80)
    db = rebuild(replace(CFG, population_size=15), BASE, cands)
    assert len(db.programs) <= 15
    assert db.best_program_id in db.programs
    assert db.programs[db.best_program_id].fitness == max(
        [BASE_FITNESS, *(c.fitness for c in cands if c.texts is not None)]
    )


def test_sampling_is_a_function_of_the_seed() -> None:
    db = rebuild(CFG, BASE, _candidates(40))

    def draw(seed: str) -> list[tuple[str, list[str]]]:
        copy_ = copy.deepcopy(db)
        copy_.rng = random.Random(seed)
        out = []
        for island in range(CFG.num_islands):
            parent, insp = copy_.sample_from_island(island, CFG.num_diverse_programs)
            out.append((parent.id, [p.id for p in insp]))
        return out

    assert draw("42:9") == draw("42:9")
    assert db.snapshot() == rebuild(CFG, BASE, _candidates(40)).snapshot()  # untouched


def test_fast_code_diversity_keeps_openevolves_constants() -> None:
    assert fast_code_diversity("a", "a") == 0.0
    # 2 characters longer (0.2), 1 more line (10), 2 characters in one set only (1.0)
    assert fast_code_diversity("ab", "ab\nc") == 0.2 + 10 + 1.0


def _attempt(number: int, speedup: float) -> Attempt:
    pair = PairTiming(0, "base_first", 0, 1.0, 1.0 / speedup, contaminated=False)
    measurement = Measurement(
        applied=True,
        tests=(SuiteResult("module", 1, 0, 0, 1.0, True), SuiteResult("full", 1, 0, 0, 1.0, True)),
        inputs=(InputTiming("dense", 1.01, "a", "a", (pair,), speedup),),
    )
    return Attempt(
        ref=AttemptRef(number, 1, 0),
        base_sha=BASE_SHA,
        patch="diff",
        prediction=None,
        rationale="",
        stop_reason=StopReason.SUBMITTED,
        usage=Usage(),
        wall_s=1.0,
        measurement=measurement,
    )


def test_fitness_is_the_speedup_of_a_real_speedup_and_zero_otherwise() -> None:
    assert records.fitness(_attempt(1, 1.8)) == 1.8
    assert records.fitness(_attempt(2, 1.0)) == 0.0  # below the floor
    assert records.fitness(replace(_attempt(3, 2.0), measurement=None)) == 0.0


def test_lineage_round_trips() -> None:
    lineage = records.Lineage(
        island=2,
        parent="0003@m1",
        parent_number=3,
        top=("0003",),
        diverse=(),
        inspirations=("base",),
        meta="m0000",
        meta_generated="Hoist.",
        texts=("a\n",),
        error="",
    )
    import json

    assert records.Lineage.from_dict(json.loads(lineage.to_line())) == lineage


def _lineage(meta_id: str, generated: str = "") -> records.Lineage:
    return records.Lineage(0, BASE_ID, 0, (), (), (), meta_id, generated, None, "")


def test_the_meta_population_scores_each_instruction_by_its_best_candidate() -> None:
    lineages = {
        1: _lineage("m0000", "Hoist lookups."),
        2: _lineage("m0000"),
        3: _lineage("m0001"),
        4: _lineage("m0000", "Use sets."),
    }
    pool = meta.population(lineages, {1: 0.0, 2: 1.4, 3: 2.2, 4: 0.0})
    assert [(p.id, p.uses, p.best) for p in pool] == [
        ("m0000", 3, 1.4),
        ("m0001", 1, 2.2),
        ("m0004", 0, None),
    ]
    assert pool[2].text == "Use sets."


def test_an_untried_instruction_is_drawn_as_often_as_the_best() -> None:
    pool = (
        meta.MetaPrompt("m0000", "", 3, 2.0),
        meta.MetaPrompt("m0003", "x", 0, None),
        meta.MetaPrompt("m0004", "y", 2, 0.0),
    )
    rng = random.Random(3)
    counts = Counter(meta.choose(pool, rng).id for _ in range(2000))
    assert counts["m0004"] < 20
    assert abs(counts["m0000"] - counts["m0003"]) < 200


def test_a_meta_reply_needs_the_tag() -> None:
    assert meta.parse_generated("sure\n<instructions>\n Look at loops.\n</instructions>") == (
        "Look at loops."
    )
    assert meta.parse_generated("no tag") == ""
