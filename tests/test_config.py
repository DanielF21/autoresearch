from pathlib import Path

import pytest

from autoresearch.config import ConfigError, load_config, parse_config

PILOT = Path(__file__).parent.parent / "configs" / "t1_w1.toml"


def test_pilot_config_loads() -> None:
    cfg = load_config(PILOT)
    assert cfg.run_id == "t1_w1"
    assert cfg.width == 1
    assert cfg.rounds == 32
    assert cfg.referee.pairs == 6
    assert cfg.referee.threshold == 1.0106
    assert cfg.referee.min_clean_pairs == 4
    assert cfg.worker.model.startswith("deepseek/")
    assert cfg.run_dir == Path("/mnt/autoresearch/runs/t1_w1")
    assert len(cfg.config_hash) == 12


def _with(key_path: str, value: str) -> str:
    """Return the pilot config with one ``section.key = value`` line replaced."""
    section, key = key_path.split(".")
    lines = PILOT.read_text().splitlines()
    out = []
    current = ""
    for line in lines:
        if line.startswith("["):
            current = line.strip("[]")
        if current == section and line.split("=")[0].strip() == key:
            out.append(f"{key} = {value}")
        else:
            out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("run.width", "0", "at least 1"),
        ("run.rounds", "true", "at least 1"),
        ("referee.threshold", "0.99", "above 1.0"),
        ("referee.min_clean_pairs", "9", "between 1 and pairs"),
        ("boxes.worker_size", '"xl"', "must be one of"),
        ("target.allow", '"networkx/**"', "list of strings"),
    ],
)
def test_invalid_values_name_the_field(key: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_with(key, value))


def test_missing_section_is_named() -> None:
    text = PILOT.read_text().replace("[storage]", "[stor]")
    with pytest.raises(ConfigError, match=r"missing \[storage\]"):
        parse_config(text)


def test_not_toml() -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        parse_config("[run\nwidth = 1")


def test_source_text_is_kept_verbatim() -> None:
    text = PILOT.read_text()
    assert parse_config(text).source_text == text
