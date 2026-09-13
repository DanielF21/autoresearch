"""Floors and docs written back into a config's text, and nothing else changed."""

from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.config import parse_config
from autoresearch.config_edit import ConfigEditError, set_docs, set_noise_floors

ROOT = Path(__file__).parent.parent
PILOT = (ROOT / "configs" / "t1_w4d.toml").read_text()

# Setups whose text looks like structure: a table header, a name key, a floor key
# and a triple quote of the other kind. None of it may be read as TOML structure.
TRICKY_TARGET = '''[target]
name = "tricky"
repo = "https://example.com/r"
sha = "abc"
package = "pkg"
alias = "p"
hot_file = "pkg/hot.py"
call = "run()"
allow = ["pkg/**"]
deny = ["tests/**"]

[target.tests]
module = "tests/test_hot.py"
full = "tests"

[[target.inputs]]
name = "first"      # a comment
setup = """
[[target.inputs]]
name = "second"
noise_floor = 9.0
x = \'\'\'not closed
run = lambda: p.go([1, 2])
"""

[[target.inputs]]
name = "second"
setup = \'\'\'
data = """ still a literal """
run = lambda: p.go(data)
\'\'\'
noise_floor = 1.5   # old
'''


def _tricky() -> str:
    head, _, rest = PILOT.partition("[target]")
    del head
    tail = rest[rest.index("[worker]") :]
    return '[run]\nrun_id = "x"\nwidth = 1\nrounds = 1\n\n' + TRICKY_TARGET + "\n" + tail


def test_floors_are_inserted_after_the_name_and_replace_an_existing_one() -> None:
    text = _tricky()
    new = set_noise_floors(text, {"first": (1.01234, "calibrated now"), "second": (1.02, "")})
    cfg = parse_config(new)
    assert [i.noise_floor for i in cfg.target.inputs] == [1.0124, 1.02]
    assert 'name = "first"      # a comment\nnoise_floor = 1.0124   # calibrated now\n' in new
    assert "# old" not in new
    # The fake header inside the first setup was left alone.
    assert cfg.target.inputs[0].setup == parse_config(text).target.inputs[0].setup


def test_every_other_line_is_kept_byte_for_byte() -> None:
    text = _tricky()
    new = set_noise_floors(text, {"first": (1.1, "")})
    added = [line for line in new.split("\n") if line not in text.split("\n")]
    assert added == ["noise_floor = 1.1000"]
    assert len(new.split("\n")) == len(text.split("\n")) + 1


def test_floors_on_the_pilot_replace_only_the_named_input() -> None:
    new = set_noise_floors(PILOT, {"gn800": (1.03, "calibrated again")})
    old, cfg = parse_config(PILOT), parse_config(new)
    assert cfg.target.inputs[3].noise_floor == 1.03
    assert replace(cfg, source_text="") == replace(
        old,
        source_text="",
        target=replace(
            old.target,
            inputs=tuple(
                replace(i, noise_floor=1.03) if i.name == "gn800" else i for i in old.target.inputs
            ),
        ),
    )


def test_an_unknown_input_is_refused() -> None:
    with pytest.raises(ConfigEditError, match="no such input: nope"):
        set_noise_floors(PILOT, {"nope": (1.1, "")})


def test_docs_replace_a_multiline_list() -> None:
    new = set_docs(PILOT, ["configs/docs/t1/a.txt", "configs/docs/t1/b.txt"])
    assert parse_config(new).target.docs == ("configs/docs/t1/a.txt", "configs/docs/t1/b.txt")
    assert "profile_er1000_005_flat" not in new
    assert "# Shown to every worker." in new


def test_docs_are_inserted_when_absent_before_the_next_table() -> None:
    new = set_docs(_tricky(), ["configs/docs/x/p.txt"])
    cfg = parse_config(new)
    assert cfg.target.docs == ("configs/docs/x/p.txt",)
    assert 'deny = ["tests/**"]\ndocs = ["configs/docs/x/p.txt"]\n' in new
