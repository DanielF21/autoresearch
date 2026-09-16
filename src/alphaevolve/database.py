"""The program database: islands and a MAP Elites grid, ported from OpenEvolve.

The paper describes its database only as combining MAP Elites with an island
model, and publishes no mechanics or parameters. Both come from OpenEvolve:
codelion/openevolve at 411fb59c886c18704caaffb611e17cf9e7d824d2,
openevolve/database.py (``ProgramDatabase``), with its iteration loop in
openevolve/process_parallel.py: a child goes to the island it was sampled for,
that island's generation counter goes up, and migration runs when the highest
counter is ``migration_interval`` past the last migration.

Kept as OpenEvolve has it: cell replacement on strictly higher fitness, the
elite archive, parent sampling by exploration, exploitation and fitness
weighted draws, inspirations from the parent's island (its best, its top, then
nearby cells, then random members), ring migration of each island's top
``migration_rate`` that never migrates a migrant twice, the population limit
that evicts programs owning no cell first, and the two feature dimensions with
min max scaling and a reference set for diversity.

Changed, each for determinism or because the harness does it elsewhere:
- a ``random.Random`` owned by the database, not the module global
- insertion ordered dicts where OpenEvolve iterates sets, whose order follows
  PYTHONHASHSEED
- one fitness number per program instead of a metrics dict
- no novelty judge (off by default there), no artifacts, no files: the run
  directory is the record and the database is rebuilt from it by ``rebuild``
- feature statistics keep the minimum and maximum, all min max scaling reads
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from alphaevolve.config import EvolveConfig
from alphaevolve.features import complexity, fast_code_diversity

BASE_ID = "base"
# The original code, 1.00x by definition. A candidate that is not a real speedup
# scores 0, below the original, as a program that fails scores in OpenEvolve.
BASE_FITNESS = 1.0
FEATURE_DIMENSIONS = ("complexity", "diversity")
DIVERSITY_CACHE_SIZE = 1000  # OpenEvolve's diversity_cache_size
WEIGHT_FLOOR = 0.001  # OpenEvolve's floor on a weight in _sample_from_island_weighted
NEARBY_RADIUS = 2  # OpenEvolve perturbs each coordinate by randint(-2, 2)
NEARBY_TRIES = 3  # and tries three times per slot to fill


@dataclass
class Program:
    """One version of the blocks. ``number`` is the attempt that produced it, 0 for
    the base; a copy or a migrant keeps its origin's number."""

    id: str
    texts: tuple[str, ...]
    fitness: float
    number: int = 0
    island: int = 0
    migrant: bool = False
    code: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.code = "".join(self.texts)


@dataclass(frozen=True)
class Candidate:
    """An attempt as the database takes it. ``texts`` is None when the attempt
    produced no program, which OpenEvolve never adds."""

    number: int
    worker: int
    texts: tuple[str, ...] | None
    fitness: float


class Database:
    def __init__(self, cfg: EvolveConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        n = cfg.num_islands
        self.programs: dict[str, Program] = {}
        self.islands: list[dict[str, None]] = [{} for _ in range(n)]
        self.island_feature_maps: list[dict[str, str]] = [{} for _ in range(n)]
        self.feature_bins = max(
            cfg.feature_bins, int(pow(cfg.archive_size, 1 / len(FEATURE_DIMENSIONS)) + 0.99)
        )
        self.current_island = 0
        self.island_generations = [0] * n
        self.last_migration_generation = 0
        self.archive: dict[str, None] = {}
        self.best_program_id: str | None = None
        self.island_best_programs: list[str | None] = [None] * n
        self.diversity_cache: dict[str, float] = {}
        self.diversity_reference_set: list[str] = []
        self.feature_stats: dict[str, tuple[float, float]] = {}
        self._copies = 0
        self._migrations = 0

    # ----- adding -----------------------------------------------------------------

    def add(self, program: Program, target_island: int | None = None) -> str:
        self.programs[program.id] = program
        coords = self._feature_coords(program)
        island = target_island if target_island is not None else self.current_island
        island %= len(self.islands)
        key = "-".join(str(c) for c in coords)
        cells = self.island_feature_maps[island]
        should_replace = key not in cells
        if not should_replace:
            existing = cells[key]
            should_replace = existing not in self.programs or self._is_better(
                program, self.programs[existing]
            )
        replaced: str | None = None
        if should_replace:
            if key in cells:
                existing = cells[key]
                if existing in self.programs and existing in self.archive:
                    del self.archive[existing]
                    self.archive[program.id] = None
                self.islands[island].pop(existing, None)
                replaced = existing
            cells[key] = program.id
        self.islands[island][program.id] = None
        program.island = island
        self._update_archive(program)
        self._enforce_population_limit(exclude=program.id)
        self._update_best(program)
        self._update_island_best(program, island)
        if replaced is not None and replaced not in (program.id, self.best_program_id):
            self._remove_if_orphaned(replaced)
        return program.id

    @staticmethod
    def _is_better(a: Program, b: Program) -> bool:
        return a.fitness > b.fitness

    def _update_archive(self, program: Program) -> None:
        size = self.cfg.archive_size
        if len(self.archive) < size:
            self.archive[program.id] = None
            return
        for pid in [p for p in self.archive if p not in self.programs]:
            del self.archive[pid]
        if len(self.archive) < size:
            self.archive[program.id] = None
            return
        valid = [self.programs[p] for p in self.archive]
        if not valid:
            self.archive[program.id] = None
            return
        worst = min(valid, key=lambda p: p.fitness)
        if self._is_better(program, worst):
            del self.archive[worst.id]
            self.archive[program.id] = None

    def _update_best(self, program: Program) -> None:
        if (
            self.best_program_id is None
            or self.best_program_id not in self.programs
            or self._is_better(program, self.programs[self.best_program_id])
        ):
            self.best_program_id = program.id

    def _update_island_best(self, program: Program, island: int) -> None:
        current = self.island_best_programs[island]
        if (
            current is None
            or current not in self.programs
            or self._is_better(program, self.programs[current])
        ):
            self.island_best_programs[island] = program.id

    # ----- features ---------------------------------------------------------------

    def _feature_coords(self, program: Program) -> list[int]:
        coords = [self._bin("complexity", float(complexity(program.code)))]
        if len(self.programs) < 2:
            coords.append(0)
        else:
            coords.append(self._bin("diversity", self._cached_diversity(program)))
        return coords

    def _bin(self, name: str, value: float) -> int:
        low, high = self.feature_stats.get(name, (value, value))
        low, high = min(low, value), max(high, value)
        self.feature_stats[name] = (low, high)
        scaled = 0.5 if high == low else min(1.0, max(0.0, (value - low) / (high - low)))
        return max(0, min(self.feature_bins - 1, int(scaled * self.feature_bins)))

    def _cached_diversity(self, program: Program) -> float:
        code = program.code
        if code in self.diversity_cache:
            return self.diversity_cache[code]
        if len(self.diversity_reference_set) < self.cfg.diversity_reference_size:
            self._update_reference_set()
        scores = [
            fast_code_diversity(code, ref) for ref in self.diversity_reference_set if ref != code
        ]
        value = sum(scores) / max(1, len(scores)) if scores else 0.0
        if len(self.diversity_cache) >= DIVERSITY_CACHE_SIZE:
            del self.diversity_cache[next(iter(self.diversity_cache))]
        self.diversity_cache[code] = value
        return value

    def _update_reference_set(self) -> None:
        everyone = list(self.programs.values())
        size = self.cfg.diversity_reference_size
        if not everyone:
            return
        if len(everyone) <= size:
            self.diversity_reference_set = [p.code for p in everyone]
            return
        remaining = everyone.copy()
        selected = [remaining.pop(self.rng.randint(0, len(remaining) - 1))]
        while len(selected) < size and remaining:
            best_value, best_index = -1.0, -1
            for i, candidate in enumerate(remaining):
                nearest = min(fast_code_diversity(candidate.code, s.code) for s in selected)
                if nearest > best_value:
                    best_value, best_index = nearest, i
            if best_index >= 0:
                selected.append(remaining.pop(best_index))
        self.diversity_reference_set = [p.code for p in selected]

    # ----- sampling ---------------------------------------------------------------

    def sample(self, n: int) -> tuple[Program, list[Program]]:
        parent = self._sample_parent()
        return parent, self._sample_inspirations(parent, n)

    def sample_from_island(self, island: int, n: int) -> tuple[Program, list[Program]]:
        island %= len(self.islands)
        if not self.islands[island]:
            return self.sample(n)
        r = self.rng.random()
        if r < self.cfg.exploration_ratio:
            parent = self._from_island_random(island)
        elif r < self.cfg.exploration_ratio + self.cfg.exploitation_ratio:
            parent = self._from_archive_for_island(island)
        else:
            parent = self._from_island_weighted(island)
        return parent, self._sample_inspirations(parent, n, island)

    def _sample_parent(self) -> Program:
        r = self.rng.random()
        if r < self.cfg.exploration_ratio:
            return self._exploration_parent()
        if r < self.cfg.exploration_ratio + self.cfg.exploitation_ratio:
            return self._exploitation_parent()
        return self._random_parent()

    def _copy_of_best(self, island: int) -> Program:
        assert self.best_program_id is not None
        best = self.programs[self.best_program_id]
        self._copies += 1
        copy = Program(
            id=f"{best.id}@copy{self._copies}",
            texts=best.texts,
            fitness=best.fitness,
            number=best.number,
            island=island,
        )
        self.programs[copy.id] = copy
        self.islands[island][copy.id] = None
        return copy

    def _exploration_parent(self) -> Program:
        island = self.current_island
        valid = [p for p in self.islands[island] if p in self.programs]
        for stale in [p for p in self.islands[island] if p not in self.programs]:
            del self.islands[island][stale]
        if not valid:
            if self.best_program_id is not None and self.best_program_id in self.programs:
                return self._copy_of_best(island)
            return next(iter(self.programs.values()))
        return self.programs[self.rng.choice(valid)]

    def _exploitation_parent(self) -> Program:
        valid = [p for p in self.archive if p in self.programs]
        for stale in [p for p in self.archive if p not in self.programs]:
            del self.archive[stale]
        if not valid:
            return self._exploration_parent()
        here = [p for p in valid if self.programs[p].island == self.current_island]
        return self.programs[self.rng.choice(here or valid)]

    def _random_parent(self) -> Program:
        if not self.programs:
            raise ValueError("no programs to sample")
        return self.programs[self.rng.choice(list(self.programs))]

    def _from_island_weighted(self, island: int) -> Program:
        ids = list(self.islands[island])
        if not ids:
            return self._random_parent()
        if len(ids) == 1:
            return self.programs.get(ids[0]) or self._random_parent()
        members = [self.programs[p] for p in ids if p in self.programs]
        if not members:
            return self.programs.get(self.rng.choice(ids)) or self._random_parent()
        weights = [max(p.fitness, WEIGHT_FLOOR) for p in members]
        total = sum(weights)
        weights = [w / total for w in weights]
        return self.rng.choices(members, weights=weights, k=1)[0]

    def _from_island_random(self, island: int) -> Program:
        valid = [p for p in self.islands[island] if p in self.programs]
        if not valid:
            return self._random_parent()
        return self.programs[self.rng.choice(valid)]

    def _from_archive_for_island(self, island: int) -> Program:
        valid = [p for p in self.archive if p in self.programs]
        if not valid:
            return self._from_island_weighted(island)
        here = [p for p in valid if self.programs[p].island == island]
        return self.programs[self.rng.choice(here or valid)]

    def _sample_inspirations(
        self, parent: Program, n: int, island: int | None = None
    ) -> list[Program]:
        island = (parent.island if island is None else island) % len(self.islands)
        ids = list(self.islands[island])
        members = [self.programs[p] for p in ids if p in self.programs]
        if not members:
            return []
        chosen: list[Program] = []
        best = self.island_best_programs[island]
        if best is not None and best != parent.id and best in self.programs:
            chosen.append(self.programs[best])
        elif best is not None and best not in self.programs:
            self.island_best_programs[island] = None
        for p in self.get_top_programs(max(1, int(n * self.cfg.elite_selection_ratio)), island):
            if p.id != parent.id and all(p.id != c.id for c in chosen):
                chosen.append(p)
        if len(members) > n and len(chosen) < n:
            slots = n - len(chosen)
            coords = self._feature_coords(parent)
            by_cell: dict[str, str] = {}
            for pid in ids:
                if pid in self.programs:
                    key = "-".join(str(c) for c in self._feature_coords(self.programs[pid]))
                    by_cell[key] = pid
            nearby: list[Program] = []
            for _ in range(slots * NEARBY_TRIES):
                moved = [
                    max(
                        0,
                        min(
                            self.feature_bins - 1,
                            c + self.rng.randint(-NEARBY_RADIUS, NEARBY_RADIUS),
                        ),
                    )
                    for c in coords
                ]
                found = by_cell.get("-".join(str(c) for c in moved))
                if (
                    found is not None
                    and found != parent.id
                    and all(found != c.id for c in chosen)
                    and all(found != c.id for c in nearby)
                    and found in self.programs
                ):
                    nearby.append(self.programs[found])
                    if len(nearby) >= slots:
                        break
            if len(chosen) + len(nearby) < n:
                excluded = {parent.id, *(c.id for c in chosen), *(c.id for c in nearby)}
                available = [p for p in ids if p not in excluded and p in self.programs]
                if available:
                    picks = self.rng.sample(
                        available, min(n - len(chosen) - len(nearby), len(available))
                    )
                    nearby.extend(self.programs[p] for p in picks)
            chosen.extend(nearby)
        return chosen[:n]

    def get_top_programs(self, n: int, island: int | None = None) -> list[Program]:
        if island is None:
            candidates = list(self.programs.values())
        else:
            candidates = [self.programs[p] for p in self.islands[island] if p in self.programs]
        return sorted(candidates, key=lambda p: p.fitness, reverse=True)[:n]

    # ----- population -------------------------------------------------------------

    def _remove_if_orphaned(self, pid: str) -> None:
        if pid not in self.programs:
            return
        if any(pid in cells.values() for cells in self.island_feature_maps):
            return
        if any(pid in island for island in self.islands):
            return
        del self.programs[pid]
        self.archive.pop(pid, None)
        self._cleanup_stale_island_bests()

    def _enforce_population_limit(self, exclude: str | None = None) -> None:
        excess = len(self.programs) - self.cfg.population_size
        if excess <= 0:
            return
        elite = {pid for cells in self.island_feature_maps for pid in cells.values()}
        protected = {self.best_program_id, exclude} - {None}
        everyone = list(self.programs.values())
        homeless = sorted(
            (p for p in everyone if p.id not in elite and p.id not in protected),
            key=lambda p: p.fitness,
        )
        owners = sorted(
            (p for p in everyone if p.id in elite and p.id not in protected),
            key=lambda p: p.fitness,
        )
        doomed = homeless[:excess]
        if len(doomed) < excess:
            doomed.extend(owners[: excess - len(doomed)])
        for p in doomed:
            self.programs.pop(p.id, None)
            for cells in self.island_feature_maps:
                for key in [k for k, v in cells.items() if v == p.id]:
                    del cells[key]
            for island in self.islands:
                island.pop(p.id, None)
            self.archive.pop(p.id, None)
        self._cleanup_stale_island_bests()

    def _cleanup_stale_island_bests(self) -> None:
        cleared = False
        for i, best in enumerate(self.island_best_programs):
            if best is not None and (best not in self.programs or best not in self.islands[i]):
                self.island_best_programs[i] = None
                cleared = True
        if not cleared:
            return
        for i, best in enumerate(self.island_best_programs):
            if best is None and self.islands[i]:
                members = [self.programs[p] for p in self.islands[i] if p in self.programs]
                if members:
                    self.island_best_programs[i] = max(members, key=lambda p: p.fitness).id

    # ----- islands ----------------------------------------------------------------

    def increment_island_generation(self, island: int) -> None:
        self.island_generations[island] += 1

    def should_migrate(self) -> bool:
        return (
            max(self.island_generations) - self.last_migration_generation
            >= self.cfg.migration_interval
        )

    def migrate_programs(self) -> None:
        count = len(self.islands)
        if count < 2:
            return
        for i in range(count):
            members = [self.programs[p] for p in self.islands[i] if p in self.programs]
            if not members:
                continue
            members.sort(key=lambda p: p.fitness, reverse=True)
            moving = members[: max(1, int(len(members) * self.cfg.migration_rate))]
            for migrant in moving:
                if migrant.migrant:
                    continue
                for target in ((i + 1) % count, (i - 1) % count):
                    if any(
                        self.programs[p].code == migrant.code
                        for p in self.islands[target]
                        if p in self.programs
                    ):
                        continue
                    self._migrations += 1
                    copy = Program(
                        id=f"{migrant.id}@m{self._migrations}",
                        texts=migrant.texts,
                        fitness=migrant.fitness,
                        number=migrant.number,
                        island=target,
                        migrant=True,
                    )
                    self.add(copy, target_island=target)
        self.last_migration_generation = max(self.island_generations)

    def snapshot(self) -> dict[str, object]:
        """Everything that decides a later sample, for comparing two databases."""
        return {
            "programs": [(p.id, p.island, p.fitness) for p in self.programs.values()],
            "islands": [list(i) for i in self.islands],
            "cells": [dict(c) for c in self.island_feature_maps],
            "archive": list(self.archive),
            "best": self.best_program_id,
            "island_best": list(self.island_best_programs),
            "generations": list(self.island_generations),
            "last_migration": self.last_migration_generation,
            "stats": dict(self.feature_stats),
            "reference": list(self.diversity_reference_set),
            "rng": self.rng.getstate(),
        }


def extend(db: Database, candidates: Iterable[Candidate]) -> None:
    """Add attempts in order, as OpenEvolve processes completed iterations."""
    for c in candidates:
        if c.texts is None:
            continue
        program = Program(id=f"{c.number:04d}", texts=c.texts, fitness=c.fitness, number=c.number)
        db.add(program, target_island=c.worker % len(db.islands))
        db.increment_island_generation(program.island)
        if db.should_migrate():
            db.migrate_programs()


def rebuild(
    cfg: EvolveConfig, base_texts: Sequence[str], candidates: Iterable[Candidate]
) -> Database:
    """The database a run's records imply: the base, then every attempt by number."""
    db = Database(cfg, random.Random(cfg.seed))
    db.add(Program(id=BASE_ID, texts=tuple(base_texts), fitness=BASE_FITNESS))
    extend(db, sorted(candidates, key=lambda c: c.number))
    return db
