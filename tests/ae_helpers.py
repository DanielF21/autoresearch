"""Shared fakes for the AlphaEvolve tests: a config, a target file, a box that serves it."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from alphaevolve import base
from alphaevolve import config as config_mod
from alphaevolve.base import EvolveState
from alphaevolve.config import EvolveRunConfig
from autoresearch import history
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, ok
from autoresearch.boxes.image import WORK_DIR
from autoresearch.boxes.protocol import CommandResult
from autoresearch.config import RunConfig
from autoresearch.orchestrator import run as run_mod
from tests.helpers import TEST_CONFIG, referee_box

ROOT = Path(__file__).parent.parent
SEARCH = "<" * 7 + " SEARCH"
DIVIDER = "=" * 7
REPLACE = ">" * 7 + " REPLACE"

HOT = "networkx/algorithms/cluster.py"
SOURCE = (
    "import math\n\n\ndef clustering(G):\n    total = 0\n    for node in G:\n"
    "        total += 1\n    return total\n"
)
FLAT = (
    "   ncalls  tottime  percall  cumtime  percall filename:lineno(function)\n"
    "      100    0.300    0.003    0.500    0.005 networkx/algorithms/cluster.py:4(clustering)\n"
    "    70228    0.008    0.000    0.008    0.000 {built-in method builtins.len}\n"
)

EVOLVE = """
[evolve]
budget_tokens = {budget}
seed = 42
population_size = 1000
archive_size = 100
num_islands = 5
migration_interval = 50
migration_rate = 0.1
elite_selection_ratio = 0.1
exploration_ratio = 0.2
exploitation_ratio = 0.7
feature_bins = 10
diversity_reference_size = 20
num_top_programs = 3
num_diverse_programs = 2
max_code_length = 10000
"""


def config_text(budget: int = 1000, evolve: str | None = None) -> str:
    """The test harness config with tracing off, plus an [evolve] section."""
    text = TEST_CONFIG.read_text()
    at = text.index("[observe]")
    text = text[:at] + text[at:].replace("enabled = true", "enabled = false", 1)
    return text + (EVOLVE.format(budget=budget) if evolve is None else evolve)


def edit_reply(search: str, replacement: str, prose: str = "Faster.") -> str:
    return f"{prose}\n{SEARCH}\n{search}\n{DIVIDER}\n{replacement}\n{REPLACE}\n"


def flat_name(cfg: RunConfig) -> str:
    return f"profile_{cfg.target.primary.name}_flat.txt"


def evolve_factory(
    sources: dict[str, str],
    speedup: float | Callable[[str], float] = 1.5,
    blob: Callable[[bytes], str] = base.blob_sha,
) -> FakeBoxFactory:
    """Referee boxes that also serve ``sources`` at the base commit."""

    def prepare(box: FakeBox, role: str) -> None:
        referee_box(box, speedup=speedup)

        def read_base(cmd: str) -> CommandResult:
            spec = re.search(r"git rev-parse '?([^' ]+)'?", cmd)
            index = re.search(r"evolve_base_(\d+)", cmd)
            assert spec and index, cmd
            data = sources[spec.group(1).split(":", 1)[1]].encode()
            box.files[f"{WORK_DIR}/evolve_base_{index.group(1)}"] = data
            return ok(blob(data) + "\n")

        box.on("evolve_base_", read_base, first=True)

    return FakeBoxFactory(prepare=prepare)


def setup_run(
    tmp_path: Path,
    *,
    width: int = 2,
    rounds: int = 5,
    budget: int = 700,
    speedup: float | Callable[[str], float] = 1.5,
) -> tuple[EvolveRunConfig, history.RunPaths, FakeBoxFactory, EvolveState]:
    parsed = config_mod.parse(config_text(budget))
    cfg = replace(
        parsed.run, width=width, rounds=rounds, target=replace(parsed.run.target, docs=())
    )
    evolve_cfg = replace(parsed, run=cfg)
    paths = run_mod.init_run(cfg, tmp_path / "run", ((flat_name(cfg), FLAT),))
    factory = evolve_factory({HOT: SOURCE}, speedup=speedup)
    state = base.prepare(paths, evolve_cfg, factory, tmp_path / "runs")
    return evolve_cfg, paths, factory, state
