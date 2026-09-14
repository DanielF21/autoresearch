from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, fail, ok
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import load_config
from autoresearch.control import deploy as ctl

ROOT = Path(__file__).parent.parent
CFG = load_config(ROOT / "configs" / "t1_w4d.toml")


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


def test_launch_starts_a_detached_run_with_the_keys_only_in_env() -> None:
    box = FakeBox()
    envs = {"SAIL_API_KEY": "sk_secret"}
    cmd = ctl.launch(CFG, "configs/t1_w4d.toml", box, envs)
    assert box.started == [(cmd, envs)]
    assert cmd.startswith(f"cd {ctl.PACKAGE_DIR} && nohup sh -c ")
    assert f"{ctl.CLI} run configs/t1_w4d.toml" in cmd
    assert "/mnt/autoresearch/runs/t1_w4d.launch.log" in cmd and cmd.endswith("&")
    # No key may reach the command line: it is visible to anything reading ps.
    assert "sk_secret" not in cmd
    cmd = ctl.launch(CFG, "configs/t1_w4d.toml", FakeBox(), {"SAIL_API_KEY": "k"}, until=1)
    assert " --until 1;" in cmd


def test_a_launched_run_is_followed_by_releasing_the_control_box() -> None:
    cmd = ctl.launch(
        CFG, "configs/t1_w4d.toml", FakeBox(box_id="sb_control"), {"SAIL_API_KEY": "k"}
    )
    release = f"; {ctl.CLI} release-control configs/t1_w4d.toml sb_control"
    # ";" and not "&&": a run that fails leaves just as idle a control box.
    assert release in cmd
    assert cmd.index(" run configs/t1_w4d.toml") < cmd.index(release)
    kept = ctl.launch(
        CFG, "configs/t1_w4d.toml", FakeBox(box_id="sb_control"), {"k": "v"}, keep_control=True
    )
    assert "release-control" not in kept


def _proc(root: Path, pid: int, *argv: str) -> None:
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_other_runs_counts_run_processes_and_not_the_launch_shell(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc(proc, 10, "python3", "-m", "autoresearch", "run", "configs/a.toml")
    _proc(proc, 11, "/v/bin/autoresearch", "run", "configs/b.toml")
    _proc(
        proc,
        12,
        "sh",
        "-c",
        "python3 -m autoresearch run configs/a.toml; "
        "python3 -m autoresearch release-control configs/a.toml sb_1",
    )
    _proc(proc, 13, "python3", "-m", "autoresearch", "release-control", "configs/a.toml", "sb_1")
    _proc(proc, 14, "python3", "-m", "autoresearch", "status", "run")
    (proc / "self").mkdir()
    assert ctl.other_runs(proc, own_pid=13) == (10, 11)
    assert ctl.other_runs(proc, own_pid=10) == (11,)
    assert ctl.other_runs(tmp_path / "no_proc", own_pid=1) == ()


def test_nothing_in_a_box_calls_the_console_script(tmp_path: Path) -> None:
    """The script is not on a box's PATH. Every in box command goes through ``-m``.

    This is what broke the first deploy: pip installed the entry point to a
    directory the shell could not see.
    """

    def prepare(box: FakeBox, role: str) -> None:
        box.on("pip install", ok("installed\n"))
        box.on("mkdir -p", ok("runs\n"))

    box = FakeBox().on("-m autoresearch status", ok(""))
    factory = FakeBoxFactory(prepare=prepare)
    ctl.deploy(CFG, factory, _repo(tmp_path), tmp_path / "staging", "control-x")
    ctl.remote_status(CFG, box)
    commands = [
        *factory.created[0].commands,
        *box.commands,
        ctl.launch_command(CFG, "configs/t1_w4d.toml"),
    ]
    for cmd in commands:
        tokens = cmd.split()
        for i, token in enumerate(tokens):
            # Bare "autoresearch" is the console script. Paths and the volume
            # name are other tokens and are not what this guards.
            if token == "autoresearch":
                assert i > 0 and tokens[i - 1] == "-m", cmd


def test_remote_status_and_fetch(tmp_path: Path) -> None:
    box = FakeBox().on("-m autoresearch status", ok("run t1_w4d: 3 of 32 rounds\n"))
    assert "3 of 32" in ctl.remote_status(CFG, box)
    box.files["/mnt/autoresearch/runs/t1_w4d/rounds.jsonl"] = b"{}\n"
    local = ctl.fetch(CFG, box, tmp_path / "runs")
    assert local == tmp_path / "runs" / "t1_w4d"
    assert (local / "rounds.jsonl").read_bytes() == b"{}\n"
