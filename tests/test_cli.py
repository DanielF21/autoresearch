from pathlib import Path

import pytest

from autoresearch import history
from autoresearch.cli import build_parser, main
from autoresearch.types import AttemptRef, Measurement, SuiteResult
from tests.helpers import diff_for, submitted

ROOT = Path(__file__).parent.parent
PILOT = ROOT / "configs" / "t1_w1.toml"


def test_no_command_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: autoresearch" in capsys.readouterr().out


def test_every_command_is_registered() -> None:
    parser = build_parser()
    text = parser.format_help()
    for name in ("status", "run", "measure", "deploy", "launch", "remote-status", "fetch"):
        assert name in text


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
            noise_floor=1.0106,
            applied=True,
            tests=(SuiteResult("module", 1, 0, 0, 1, True), SuiteResult("full", 1, 0, 0, 1, True)),
            base_fp="a",
            patched_fp="a",
            median_ratio=1.03,
        ),
    )
    assert main(["status", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "run t1_w1: 0 of 32 rounds, 1 attempts" in out
    assert "real speedups 1" in out and "best real speedup so far 1.0300 (attempt 0001)" in out


def test_status_from_config_with_runs_root_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "t1_w1").mkdir()
    assert main(["status", str(PILOT), "--runs-root", str(tmp_path)]) == 0
    assert "0 attempts" in capsys.readouterr().out
