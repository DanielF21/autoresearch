"""Launching an AlphaEvolve run on the control box, and keeping that box alive for it."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox
from autoresearch.config import load_config
from autoresearch.control import deploy as ctl
from tests.ae_helpers import ROOT

CFG = load_config(ROOT / "configs" / "alphaevolve" / "pyparsing_ae_w16.toml")
PATH = "configs/alphaevolve/pyparsing_ae_w16.toml"


def test_the_run_step_is_alphaevolves_and_release_stays_the_harnesss() -> None:
    cmd = ctl.launch(
        CFG, PATH, FakeBox(box_id="sb_control"), {"SAIL_API_KEY": "k"}, program="alphaevolve"
    )
    assert f"{ctl.PYTHON} -m alphaevolve run {PATH} --repo-root {ctl.PACKAGE_DIR}" in cmd
    assert f"; {ctl.CLI} release-control {PATH} sb_control" in cmd
    assert "-m autoresearch run" not in cmd
    assert ctl.launch_command(CFG, PATH) == ctl.launch_command(CFG, PATH, program="autoresearch")
    with pytest.raises(ValueError, match="no run command"):
        ctl.launch_command(CFG, PATH, program="elsewhere")


def _proc(root: Path, pid: int, *argv: str) -> None:
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_an_alphaevolve_run_keeps_the_control_box_but_its_shell_does_not(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 20, "python3", "-m", "alphaevolve", "run", PATH)
    _proc(
        proc,
        21,
        "sh",
        "-c",
        f"python3 -m alphaevolve run {PATH}; python3 -m autoresearch release-control {PATH} sb",
    )
    _proc(proc, 22, "python3", "-m", "alphaevolve", "curve", "runs/a")
    assert ctl.other_runs(proc, own_pid=1) == (20,)
