from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.config import (
    LEGACY_NETWORKX_TARGET,
    ConfigError,
    SuitePaths,
    TargetSpec,
    load_config,
    parse_config,
)

CONFIGS = Path(__file__).parent.parent / "configs"
PILOT = CONFIGS / "t1_w4d.toml"  # the target agnostic shape
LEGACY = CONFIGS / "t1_w1.toml"  # the shape from before the target was described in full


def test_pilot_config_loads() -> None:
    cfg = load_config(PILOT)
    assert cfg.run_id == "t1_w4d"
    assert cfg.width == 4
    assert cfg.rounds == 32
    assert cfg.referee.pairs == 6
    assert cfg.referee.min_clean_pairs == 4
    assert cfg.referee.timing_retries == 2  # the default; the frozen config predates the key
    assert [i.name for i in cfg.target.inputs] == [
        "er1000_005",
        "er1000_001",
        "er1000_0002",
        "gn800",
        "er1600_002",
    ]
    assert cfg.target.primary.name == "er1000_005"
    assert cfg.target.primary.noise_floor == 1.0106
    assert all(i.noise_floor is not None and i.noise_floor > 1.0 for i in cfg.target.inputs)
    assert all(i.setup for i in cfg.target.inputs)
    assert cfg.target.uncalibrated == ()
    assert cfg.target.package == "networkx" and cfg.target.alias == "nx"
    assert cfg.target.package_root == "."
    assert cfg.target.pip == ("numpy", "scipy", "pandas") and cfg.target.apt == ()
    assert cfg.target.tests == SuitePaths(
        module="networkx/algorithms/tests/test_cluster.py", full="networkx"
    )
    assert cfg.target.fingerprint == ""
    assert cfg.worker.model.startswith("deepseek/")
    assert cfg.run_dir == Path("/mnt/autoresearch/runs/t1_w4d")
    assert len(cfg.config_hash) == 12


def test_a_legacy_config_reads_as_the_same_target() -> None:
    """The frozen config.toml in every existing run directory has the old shape.

    It is never rewritten, so it has to parse to the target it always meant:
    networkx bound to nx, one expression per input bound to G, the module suite
    from test_file, the full suite from the first segment of hot_file, and the
    three packages the image used to install for everyone.
    """
    legacy = load_config(CONFIGS / "t1_w4c.toml").target
    new = load_config(PILOT).target
    assert legacy == new
    assert legacy.package == LEGACY_NETWORKX_TARGET["package"]
    assert legacy.inputs[0].setup == "G = nx.erdos_renyi_graph(1000, 0.05, seed=42, directed=True)"
    # Every old config, not only the one whose run is resumable.
    for name in ("t1_w1", "t1_w4", "t1_w16", "t1_w4b"):
        old = load_config(CONFIGS / f"{name}.toml").target
        assert old.package == "networkx" and old.tests.full == "networkx"
        assert all(i.setup.startswith("G = ") for i in old.inputs)


def test_every_config_on_disk_loads() -> None:
    for path in sorted(CONFIGS.glob("*.toml")):
        load_config(path)


def test_the_frozen_config_of_the_resumable_run_is_byte_identical() -> None:
    """runs/t1_w4c is resumed against configs/t1_w4c.toml by byte comparison."""
    frozen = Path(__file__).parent.parent / "runs" / "t1_w4c" / "config.toml"
    if not frozen.exists():
        pytest.skip("runs/t1_w4c is not on this machine")
    assert frozen.read_bytes() == (CONFIGS / "t1_w4c.toml").read_bytes()


def _with(key_path: str, value: str, source: Path = PILOT) -> str:
    """Return a config with one ``section.key = value`` line replaced."""
    section, key = key_path.split(".")
    lines = source.read_text().splitlines()
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


def test_timing_retries_is_read_and_validated() -> None:
    # The frozen configs predate the key, so it is added beside min_clean_pairs here.
    text = PILOT.read_text().replace(
        "min_clean_pairs = 4", "min_clean_pairs = 4\ntiming_retries = 3"
    )
    assert parse_config(text).referee.timing_retries == 3
    with pytest.raises(ConfigError, match="at least 0"):
        parse_config(text.replace("timing_retries = 3", "timing_retries = -1"))
    with pytest.raises(ConfigError, match="at least 0"):
        parse_config(text.replace("timing_retries = 3", "timing_retries = true"))


def test_prompt_versions_default_to_v1_and_are_checked_by_name() -> None:
    assert load_config(PILOT).worker.prompts == ("v1",)
    text = PILOT.read_text().replace(
        "max_turns = 80", 'max_turns = 80\nprompts = ["v1", "v2", "v3", "v4"]'
    )
    assert parse_config(text).worker.prompts == ("v1", "v2", "v3", "v4")
    with pytest.raises(ConfigError, match=r"unknown versions \['v9'\]; known: \['v1'"):
        parse_config(text.replace('"v4"', '"v9"'))
    with pytest.raises(ConfigError, match="non empty list"):
        parse_config(text.replace('["v1", "v2", "v3", "v4"]', "[]"))
    four = load_config(CONFIGS / "t1_p3_w4.toml")
    assert four.worker.prompts == ("v1", "v2", "v3", "v4") and four.width == 4


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("run.width", "0", "at least 1"),
        ("run.rounds", "true", "at least 1"),
        ("referee.min_clean_pairs", "9", "between 1 and pairs"),
        ("boxes.worker_size", '"xl"', "must be one of"),
        ("target.allow", '"networkx/**"', "list of strings"),
        ("target.alias", '"not an identifier"', "Python identifier"),
        ("target.package_root", '""', "non empty"),
    ],
)
def test_invalid_values_name_the_field(key: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_with(key, value))


def _target_block(*inputs: str, source: Path = PILOT) -> str:
    """The config with its [[target.inputs]] tables replaced."""
    text = source.read_text()
    head, _, tail = text.partition("[[target.inputs]]")
    return head + "\n\n".join(inputs) + "\n\n[worker]" + tail.partition("[worker]")[2]


def test_a_target_needs_at_least_one_input() -> None:
    text = PILOT.read_text()
    head, _, tail = text.partition("[[target.inputs]]")
    with pytest.raises(ConfigError, match="missing 'inputs'"):
        parse_config(head + "[worker]" + tail.partition("[worker]")[2])
    # Present but empty is a different mistake and gets its own message. The
    # key has to sit in [target] itself, above the [target.tests] table.
    empty = head.replace("[target]\n", "[target]\ninputs = []\n", 1)
    with pytest.raises(ConfigError, match="at least once"):
        parse_config(empty + "[worker]" + tail.partition("[worker]")[2])


def test_input_names_must_be_unique() -> None:
    one = '[[target.inputs]]\nname = "a"\nsetup = "G = nx.g()"\nnoise_floor = 1.02\n'
    with pytest.raises(ConfigError, match="appears twice"):
        parse_config(_target_block(one, one))


def test_an_input_floor_must_be_above_one() -> None:
    bad = '[[target.inputs]]\nname = "a"\nsetup = "G = nx.g()"\nnoise_floor = 0.99\n'
    with pytest.raises(ConfigError, match=r"above 1\.0"):
        parse_config(_target_block(bad))


def test_an_input_needs_a_setup() -> None:
    bad = '[[target.inputs]]\nname = "a"\nnoise_floor = 1.02\n'
    with pytest.raises(ConfigError, match="setup"):
        parse_config(_target_block(bad))


def test_an_input_without_a_floor_is_uncalibrated_not_invalid() -> None:
    """A new target has no floors until autoresearch calibrate has run. The config must
    load so calibrate can read it; a run is what refuses it, by name."""
    floored = '[[target.inputs]]\nname = "a"\nsetup = "G = nx.g()"\nnoise_floor = 1.02\n'
    bare = '[[target.inputs]]\nname = "b"\nsetup = "G = nx.h()"\n'
    cfg = parse_config(_target_block(floored, bare))
    assert [i.noise_floor for i in cfg.target.inputs] == [1.02, None]
    assert cfg.target.uncalibrated == ("b",)


def test_the_old_input_spelling_is_refused_once_the_package_is_named() -> None:
    """``graph`` is read only through the legacy table. With ``package`` set the
    config is in the new shape and the old key is a mistake, named as such."""
    old = '[[target.inputs]]\nname = "a"\ngraph = "nx.g()"\nnoise_floor = 1.02\n'
    with pytest.raises(ConfigError, match="old spelling"):
        parse_config(_target_block(old))


def test_the_old_test_file_key_is_refused_once_the_package_is_named() -> None:
    text = PILOT.read_text().replace(
        'package = "networkx"', 'package = "networkx"\ntest_file = "x.py"', 1
    )
    with pytest.raises(ConfigError, match=r"moved to \[target.tests\]"):
        parse_config(text)


def test_a_legacy_config_cannot_mix_in_new_keys() -> None:
    text = LEGACY.read_text().replace(
        "[target]\n", "[target]\ntests = { module = 'a', full = 'b' }\n", 1
    )
    with pytest.raises(ConfigError, match="name the package"):
        parse_config(text)


def test_the_new_shape_requires_both_test_paths() -> None:
    text = PILOT.read_text().replace('full = "networkx"', "", 1)
    with pytest.raises(ConfigError, match=r"\[target.tests\] is missing 'full'"):
        parse_config(text)


def test_optional_target_keys_default_to_empty() -> None:
    text = PILOT.read_text().replace('pip = ["numpy", "scipy", "pandas"]', "", 1)
    cfg = parse_config(text)
    assert cfg.target.pip == () and cfg.target.apt == ()
    assert cfg.target.fingerprint == ""
    withfp = PILOT.read_text().replace(
        "[target]\n", '[target]\nfingerprint = "sorted(result)"\n', 1
    )
    assert parse_config(withfp).target.fingerprint == "sorted(result)"


def test_target_spec_round_trips_to_a_dict_with_no_target_specific_keys() -> None:
    spec = TargetSpec(
        name="t",
        repo="https://example.invalid/r",
        sha="abc",
        package="pkg",
        alias="p",
        package_root="src",
        hot_file="src/pkg/hot.py",
        call="p.f(x)",
        tests=SuitePaths("tests/test_hot.py", "tests"),
        inputs=(),
        allow=("src/**",),
        deny=("tests/**",),
        pip=("dep",),
        apt=("libfoo",),
        fingerprint="len(result)",
    )
    d = spec.to_dict()
    assert d["tests"] == {"module": "tests/test_hot.py", "full": "tests"}
    assert d["package_root"] == "src" and d["pip"] == ["dep"] and d["apt"] == ["libfoo"]
    assert "graph" not in str(d) and "test_file" not in d
    assert replace(spec, inputs=()).uncalibrated == ()


def test_the_old_referee_noise_floor_key_is_refused_by_name() -> None:
    """A config from before the change must not load as if it still made sense."""
    text = PILOT.read_text().replace("pairs = 6", "pairs = 6\nnoise_floor = 1.0106", 1)
    with pytest.raises(ConfigError, match=r"moved to \[\[target.inputs\]\]"):
        parse_config(text)


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
