from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory, fail, ok
from autoresearch.boxes.image import REPO_DIR, clone_commands
from autoresearch.boxes.protocol import BoxError, CommandResult
from autoresearch.config import load_config

PILOT = Path(__file__).parent.parent / "configs" / "t1_w1.toml"


def test_fake_box_answers_by_substring_and_records_commands() -> None:
    box = FakeBox().on("pytest", ok('{"ok": true}')).on("git", fail("no repo"))
    assert box.run("python -m pytest x", timeout=1).stdout == '{"ok": true}'
    assert box.run("git status", timeout=1).exit_code == 1
    assert box.run("unknown", timeout=1).exit_code == 127
    assert box.commands == ["python -m pytest x", "git status", "unknown"]


def test_fake_box_callable_handler_sees_the_command() -> None:
    box = FakeBox().on("echo", lambda cmd: ok(cmd.split(" ", 1)[1]))
    assert box.run("echo hello", timeout=1).stdout == "hello"


def test_fake_box_files_and_upload(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("print(2)\n")
    box = FakeBox()
    box.write("/w/x.txt", b"hi")
    assert box.read("/w/x.txt") == b"hi"
    box.upload_dir(tmp_path, "/workspace/guest")
    assert box.read("/workspace/guest/a.py") == b"print(1)\n"
    assert box.read("/workspace/guest/sub/b.py") == b"print(2)\n"
    with pytest.raises(BoxError):
        box.read("/missing")


def test_fake_box_refuses_commands_after_terminate() -> None:
    box = FakeBox()
    box.terminate()
    with pytest.raises(BoxError):
        box.run("ls", timeout=1)


def test_fake_factory_prepares_and_reattaches() -> None:
    seen: list[str] = []

    def prepare(box: FakeBox, role: str) -> None:
        seen.append(role)
        box.on("ls", ok("prepared"))

    factory = FakeBoxFactory(prepare=prepare)
    box = factory.create(name="ref-0", role="referee")
    assert seen == ["referee"]
    assert box.run("ls", timeout=1).stdout == "prepared"
    assert factory.reattach(box.box_id) is box
    box.terminate()
    assert factory.reattach(box.box_id) is None


def test_command_result_last_json_line() -> None:
    r = CommandResult(0, 'noise\n{"a": 1}\nmore\n{"b": 2}\n', "")
    assert r.last_json_line() == '{"b": 2}'
    assert CommandResult(0, "nothing", "").last_json_line() is None
    assert not CommandResult(0, "", "", timed_out=True).ok


def test_image_clone_commands_pin_the_sha() -> None:
    target = load_config(PILOT).target
    cmds = clone_commands(target)
    assert cmds[0].startswith(f"git clone -q {target.repo} {REPO_DIR}")
    assert f"git checkout -q {target.sha}" in cmds[1]
