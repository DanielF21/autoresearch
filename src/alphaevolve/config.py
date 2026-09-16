"""An AlphaEvolve run's config: a harness config plus a strict ``[evolve]`` section.

Every section the harness reads means what it means for a harness run: the
target, the referee, the boxes, the storage, and the model settings in
``[worker]``. ``[run].width`` is the candidates per batch and the referee boxes
held, and ``[run].rounds`` is a hard cap on batches; the token budget is what
normally ends a run.

``[evolve]`` has no defaults. Every parameter is written out in the config file
beside its source, so a run can never quietly use a value nobody chose. Unknown
keys are refused, since a misspelt parameter would otherwise be ignored.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.config import ConfigError, RunConfig, parse_config

INT_KEYS = (
    "seed",
    "population_size",
    "archive_size",
    "num_islands",
    "migration_interval",
    "feature_bins",
    "diversity_reference_size",
    "num_top_programs",
    "num_diverse_programs",
    "max_code_length",
)
RATIO_KEYS = (
    "migration_rate",
    "elite_selection_ratio",
    "exploration_ratio",
    "exploitation_ratio",
)
BUDGET_KEYS = ("budget_tokens", "budget_from_run")


@dataclass(frozen=True)
class EvolveConfig:
    """The ``[evolve]`` section. The budget is one of ``budget_tokens``, a count of
    prompt plus completion tokens, or ``budget_from_run``, the run id of a harness
    run under the same runs root whose total becomes the budget."""

    seed: int
    population_size: int
    archive_size: int
    num_islands: int
    migration_interval: int
    feature_bins: int
    diversity_reference_size: int
    num_top_programs: int
    num_diverse_programs: int
    max_code_length: int
    migration_rate: float
    elite_selection_ratio: float
    exploration_ratio: float
    exploitation_ratio: float
    budget_tokens: int | None
    budget_from_run: str


@dataclass(frozen=True)
class EvolveRunConfig:
    run: RunConfig
    evolve: EvolveConfig


def _int(section: dict[str, Any], key: str) -> int:
    value = section[key]
    low = 0 if key == "seed" else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ConfigError(f"[evolve].{key} must be an integer of at least {low}, got {value!r}")
    return value


def _ratio(section: dict[str, Any], key: str) -> float:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"[evolve].{key} must be a number between 0 and 1, got {value!r}")
    return float(value)


def parse(text: str) -> EvolveRunConfig:
    """Parse and validate. Raises ConfigError naming the field."""
    run = parse_config(text)
    section = tomllib.loads(text).get("evolve")
    if not isinstance(section, dict):
        raise ConfigError("missing [evolve] section")
    unknown = sorted(set(section) - {*INT_KEYS, *RATIO_KEYS, *BUDGET_KEYS})
    if unknown:
        raise ConfigError(f"[evolve] has unknown keys {unknown}")
    missing = [k for k in (*INT_KEYS, *RATIO_KEYS) if k not in section]
    if missing:
        raise ConfigError(
            f"[evolve] is missing {missing}; every parameter is written out with its source"
        )
    ints = {k: _int(section, k) for k in INT_KEYS}
    ratios = {k: _ratio(section, k) for k in RATIO_KEYS}
    if ratios["exploration_ratio"] + ratios["exploitation_ratio"] > 1.0:
        raise ConfigError("[evolve].exploration_ratio plus exploitation_ratio must be at most 1")

    if ("budget_tokens" in section) == ("budget_from_run" in section):
        raise ConfigError("[evolve] needs exactly one of budget_tokens and budget_from_run")
    budget_tokens: int | None = None
    budget_from_run = ""
    if "budget_tokens" in section:
        budget_tokens = _int(section, "budget_tokens")
    else:
        raw = section["budget_from_run"]
        if not isinstance(raw, str) or not raw:
            raise ConfigError("[evolve].budget_from_run must be a harness run id")
        budget_from_run = raw

    if run.worker.hidden_slots:
        raise ConfigError("[worker].hidden_slots must be 0: every candidate reads the database")
    return EvolveRunConfig(
        run=run,
        evolve=EvolveConfig(
            seed=ints["seed"],
            population_size=ints["population_size"],
            archive_size=ints["archive_size"],
            num_islands=ints["num_islands"],
            migration_interval=ints["migration_interval"],
            feature_bins=ints["feature_bins"],
            diversity_reference_size=ints["diversity_reference_size"],
            num_top_programs=ints["num_top_programs"],
            num_diverse_programs=ints["num_diverse_programs"],
            max_code_length=ints["max_code_length"],
            migration_rate=ratios["migration_rate"],
            elite_selection_ratio=ratios["elite_selection_ratio"],
            exploration_ratio=ratios["exploration_ratio"],
            exploitation_ratio=ratios["exploitation_ratio"],
            budget_tokens=budget_tokens,
            budget_from_run=budget_from_run,
        ),
    )


def load(path: Path) -> EvolveRunConfig:
    return parse(path.read_text())
