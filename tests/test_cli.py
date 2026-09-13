from pathlib import Path

import pytest

from autoresearch import env, history
from autoresearch.cli import build_parser, main
from autoresearch.types import AttemptRef, InputTiming, Measurement, SuiteResult
from tests.helpers import diff_for, submitted

ROOT = Path(__file__).parent.parent
PILOT = ROOT / "configs" / "t1_w4d.toml"


def test_no_command_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: autoresearch" in capsys.readouterr().out


def test_every_command_is_registered() -> None:
    parser = build_parser()
    text = parser.format_help()
    for name in (
        "status",
        "run",
        "measure",
        "check",
        "reap",
        "deploy",
        "launch",
        "remote-status",
        "fetch",
        "release-control",
    ):
        assert name in text


def test_reap_lists_and_terminates_only_with_yes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoresearch.boxes import sail_box

    live = [sail_box.LiveBox("sb_1", "referee-x-0", "running", "2026-09-13 02:28:51+00:00")]
    ended: list[str] = []
    monkeypatch.setattr(env, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(sail_box, "live_boxes", lambda prefix="": live)
    monkeypatch.setattr(sail_box, "terminate_box", ended.append)
    assert main(["reap"]) == 0
    assert "referee-x-0" in capsys.readouterr().out and ended == []
    assert main(["reap", "--yes"]) == 0
    assert ended == ["sb_1"] and "terminated referee-x-0" in capsys.readouterr().out


def test_run_and_measure_refuse_an_uncalibrated_config_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A floorless input is fine to check or calibrate and never to run. Refused
    before any box or model is touched."""
    text = PILOT.read_text().replace("noise_floor = 1.0218", "", 1)  # gn800's floor
    cfg = tmp_path / "bare.toml"
    cfg.write_text(text)
    assert main(["run", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "cannot run" in err and "gn800" in err and "autoresearch calibrate" in err
    assert main(["measure", str(cfg), "x.diff"]) == 2
    assert "cannot measure" in capsys.readouterr().err


def test_status_on_a_run_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "t1_w1"
    run_dir.mkdir()
    (run_dir / history.CONFIG_FILE).write_text(PILOT.read_text())
    paths = history.RunPaths(run_dir)
    ref = AttemptRef(1, 1, 0)
    history.write_attempt(paths, ref, "sha", {}, submitted(diff_for("a")), "")
    history.write_measurement(
        paths,
        ref,
        Measurement(
            applied=True,
            tests=(SuiteResult("module", 1, 0, 0, 1, True), SuiteResult("full", 1, 0, 0, 1, True)),
            inputs=(
                InputTiming(
                    name="bench", noise_floor=1.0106, base_fp="a", patched_fp="a", speedup=1.03
                ),
            ),
        ),
    )
    assert main(["status", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "run t1_w4d: 0 of 32 rounds, 1 attempts" in out
    assert "real speedups 1" in out and "best real speedup so far 1.0300 (attempt 0001)" in out


def test_status_from_config_with_runs_root_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "t1_w1").mkdir()
    assert main(["status", str(PILOT), "--runs-root", str(tmp_path)]) == 0
    assert "0 attempts" in capsys.readouterr().out
