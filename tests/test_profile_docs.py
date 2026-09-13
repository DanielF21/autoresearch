"""Profiling on the referee, and the documents a worker is shown from it."""

from pathlib import Path

from autoresearch.config import load_config
from autoresearch.referee import profile_docs
from autoresearch.referee.referee import InputProfile, Referee
from tests.helpers import referee_box

ROOT = Path(__file__).parent.parent
CFG = load_config(ROOT / "configs" / "t1_w4d.toml")


def test_profile_runs_every_input_on_the_base_tree_and_cleans_up() -> None:
    box = referee_box()
    ref = Referee(box, CFG)
    ref.setup()
    rows = ref.profile()
    assert [r.name for r in rows] == [i.name for i in CFG.target.inputs]
    assert rows[0].setup == CFG.target.inputs[0].setup and rows[0].share == 0.8
    launches = [c for c in box.commands if "time_target.py" in c]
    assert len(launches) == len(CFG.target.inputs)
    assert all("--profile" in c and "/workspace/work/base" in c for c in launches)
    assert not any("--patch" in c or "run_tests.py" in c for c in box.commands)
    assert sum(1 for c in box.commands if "--remove" in c) == 2


def test_documents_are_named_after_the_primary_input() -> None:
    rows = [
        InputProfile("big", "G = nx.big()", 2.0, 1.5, 0.75, "FLAT", "CALLERS"),
        InputProfile("small", "G = nx.small()", 0.02, 0.01, 0.5, "", ""),
    ]
    docs = profile_docs.documents(CFG.target, rows)
    assert list(docs) == ["profile_inputs.txt", "profile_big_flat.txt", "profile_big_callers.txt"]
    table = docs["profile_inputs.txt"]
    assert "big" in table and "75.0%" in table and "spans 100x between big and small" in table
    assert "geometric mean of the hot file's share: 61.2%" in table
    assert docs["profile_big_flat.txt"].endswith("FLAT")
    assert "G = nx.big()" in docs["profile_big_callers.txt"]
