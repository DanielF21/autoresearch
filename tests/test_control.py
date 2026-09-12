from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, fail, ok
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import load_config
from autoresearch.control import deploy as ctl

ROOT = Path(__file__).parent.parent
CFG = load_config(ROOT / "configs" / "t1_w1.toml")


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src" / "autoresearch").mkdir(parents=True)
    (root / "src" / "autoresearch" / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / ".venv").mkdir()
    (root / ".venv" / "big").write_text("no")
    (root / "runs").mkdir()
    (root / "runs" / "secret.jsonl").write_text("no")
    return root


def test_deploy_uploads_the_package_without_venv_or_runs(tmp_path: Path) -> None:
    def prepare(box: FakeBox, role: str) -> None:
        assert role == "control"
        box.on("pip install", ok("installed\n"))
        box.on("mkdir -p", ok("runs\n"))

    factory = FakeBoxFactory(prepare=prepare)
    info = ctl.deploy(CFG, factory, _repo(tmp_path), tmp_path / "staging", "control-x")
    box = factory.created[0]
    assert info.box_id == box.box_id and info.mount == "/mnt/autoresearch"
    assert box.files[f"{ctl.PACKAGE_DIR}/pyproject.toml"] == b"[project]\nname='x'\n"
    assert not any(
        ".venv" in p or "/runs/" in p for p in box.files if p.startswith(ctl.PACKAGE_DIR)
    )
    assert box.files["/mnt/autoresearch/.volume"] == b"autoresearch"
    assert not box.terminated


def test_deploy_terminates_the_box_when_install_fails(tmp_path: Path) -> None:
    def prepare(box: FakeBox, role: str) -> None:
        box.on("pip install", fail("no pip"))

    factory = FakeBoxFactory(prepare=prepare)
    with pytest.raises(BoxError, match="install failed"):
        ctl.deploy(CFG, factory, _repo(tmp_path), tmp_path / "staging", "control-x")
    assert factory.created[0].terminated


def test_control_record_round_trip(tmp_path: Path) -> None:
    record = tmp_path / "control.json"
    info = ctl.ControlInfo("sb_1", "control-x", "autoresearch", "/mnt/autoresearch")
    ctl.save_control(info, record)
    assert ctl.load_control(record) == info
    with pytest.raises(BoxError, match="deploy first"):
        ctl.load_control(tmp_path / "missing.json")


def test_launch_starts_a_detached_run_with_the_key_only_in_env() -> None:
    box = FakeBox()
    cmd = ctl.launch(CFG, "configs/t1_w1.toml", box, "sk_secret")
    assert box.started == [(cmd, {"SAIL_API_KEY": "sk_secret"})]
    assert cmd.startswith(f"cd {ctl.PACKAGE_DIR} && nohup autoresearch run configs/t1_w1.toml")
    assert "/mnt/autoresearch/runs/t1_w1.launch.log" in cmd and cmd.endswith("&")
    assert "sk_secret" not in cmd
    cmd = ctl.launch(CFG, "configs/t1_w1.toml", FakeBox(), "k", until=1)
    assert " --until 1 >>" in cmd


def test_remote_status_and_fetch(tmp_path: Path) -> None:
    box = FakeBox().on("autoresearch status", ok("run t1_w1: 3 of 32 rounds\n"))
    assert "3 of 32" in ctl.remote_status(CFG, box)
    box.files["/mnt/autoresearch/runs/t1_w1/rounds.jsonl"] = b"{}\n"
    local = ctl.fetch(CFG, box, tmp_path / "runs")
    assert local == tmp_path / "runs" / "t1_w1"
    assert (local / "rounds.jsonl").read_bytes() == b"{}\n"
