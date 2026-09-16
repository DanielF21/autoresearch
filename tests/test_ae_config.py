"""The AlphaEvolve config: a harness config the harness still reads, plus a strict [evolve]."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from alphaevolve import config
from autoresearch.config import ConfigError, load_config
from tests.ae_helpers import EVOLVE, ROOT, config_text

AE = ROOT / "configs" / "alphaevolve"


@pytest.mark.parametrize("repo", ["pyparsing", "pycodestyle"])
def test_the_run_configs_parse_and_share_the_harness_arms_target(repo: str) -> None:
    path = AE / f"{repo}_ae_w16.toml"
    parsed = config.load(path)
    harness = load_config(ROOT / "configs" / f"{repo}_control_of_w16.toml")
    assert parsed.run.run_id == f"{repo}_ae_w16"
    # The harness's own commands (status, fetch, remote-status) read the same file.
    assert load_config(path).run_id == f"{repo}_ae_w16"
    assert parsed.run.target == harness.target
    # The same referee, plus the two early stops that only AlphaEvolve sets.
    assert parsed.run.referee.measurement_timeout == 600
    assert parsed.run.referee.stop_on_failed_tests is True
    own = replace(parsed.run.referee, measurement_timeout=None, stop_on_failed_tests=False)
    assert own == harness.referee
    # Switched from the harness's model to GLM 5.3 part way through, after endpoint
    # timeouts; each run folder's model_switch.json names the first GLM round.
    assert harness.worker.model == "deepseek/deepseek-v4-pro-0813"
    assert parsed.run.worker.model == "zai-org/GLM-5.3"
    assert parsed.run.worker.reasoning_effort == harness.worker.reasoning_effort
    assert parsed.evolve.budget_tokens == 1_000_000_000 and parsed.run.rounds == 64
    assert parsed.evolve.budget_from_run == ""
    assert parsed.run.observe.enabled


def test_every_parameter_is_required_and_unknown_keys_are_refused() -> None:
    parsed = config.parse(config_text(500))
    assert parsed.evolve.budget_tokens == 500 and parsed.evolve.num_islands == 5
    missing = EVOLVE.format(budget=500).replace("num_islands = 5\n", "")
    with pytest.raises(ConfigError, match="missing"):
        config.parse(config_text(evolve=missing))
    extra = EVOLVE.format(budget=500) + "num_isladns = 5\n"
    with pytest.raises(ConfigError, match="unknown keys"):
        config.parse(config_text(evolve=extra))
    with pytest.raises(ConfigError, match="missing \\[evolve\\]"):
        config.parse(config_text(evolve=""))


def test_exactly_one_budget() -> None:
    both = EVOLVE.format(budget=500) + 'budget_from_run = "x"\n'
    with pytest.raises(ConfigError, match="exactly one"):
        config.parse(config_text(evolve=both))
    neither = re.sub(r"budget_tokens = \d+\n", "", EVOLVE.format(budget=500))
    with pytest.raises(ConfigError, match="exactly one"):
        config.parse(config_text(evolve=neither))
    named = neither + 'budget_from_run = "pyparsing_control_of_w16"\n'
    assert config.parse(config_text(evolve=named)).evolve.budget_from_run == (
        "pyparsing_control_of_w16"
    )


def test_ratios_are_checked() -> None:
    bad = EVOLVE.format(budget=500).replace("migration_rate = 0.1", "migration_rate = 1.5")
    with pytest.raises(ConfigError, match="migration_rate"):
        config.parse(config_text(evolve=bad))
    over = EVOLVE.format(budget=500).replace("exploitation_ratio = 0.7", "exploitation_ratio = 0.9")
    with pytest.raises(ConfigError, match="at most 1"):
        config.parse(config_text(evolve=over))


def test_hidden_slots_are_refused_and_tracing_is_allowed() -> None:
    hidden = config_text(500).replace("[worker]\n", "[worker]\nhidden_slots = 1\n", 1)
    with pytest.raises(ConfigError, match="hidden_slots"):
        config.parse(hidden)
    traced = config_text(500)
    at = traced.index("[observe]")
    traced = traced[:at] + traced[at:].replace("enabled = false", "enabled = true", 1)
    assert config.parse(traced).run.observe.enabled
