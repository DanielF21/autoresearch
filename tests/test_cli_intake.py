"""The intake, profile, calibrate and next commands, wired end to end against fakes."""

import json
import subprocess
from pathlib import Path

import pytest

from autoresearch import env
from autoresearch.boxes.fake_box import FakeBox, FakeBoxFactory
from autoresearch.cli import build_parser, main
from autoresearch.config import load_config, parse_config
from autoresearch.config_edit import set_docs
from autoresearch.model.fake_model import FakeChatModel, tool_call
from autoresearch.referee import admissibility
from tests.helpers import referee_box
from tests.test_intake import GOOD, _git_repo

ROOT = Path(__file__).parent.parent
PILOT = ROOT / "configs" / "t1_w4d.toml"
GN800_FLOOR = "noise_floor = 1.0218          # calibrated, 42 clean pairs, null sd 0.0214"


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "load_dotenv", lambda *a, **k: None)


def _fake_boxes(
    monkeypatch: pytest.MonkeyPatch, prepare: object = None, speedup: float = 1.01
) -> FakeBoxFactory:
    from autoresearch.boxes import sail_box

    def setup(box: FakeBox, role: str) -> None:
        referee_box(box, speedup=speedup)
        if callable(prepare):
            prepare(box, role)

    factory = FakeBoxFactory(prepare=setup)
    monkeypatch.setattr(sail_box, "SailBoxFactory", lambda cfg: factory)
    return factory


def _config(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "t.toml"
    path.write_text(PILOT.read_text() if text is None else text)
    return path


def test_the_new_commands_are_registered() -> None:
    help_text = build_parser().format_help()
    for name in ("intake", "profile", "calibrate", "next"):
        assert name in help_text


# ----- intake ------------------------------------------------------------------------


def test_intake_drafts_a_repository_and_names_the_model_step_next(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _git_repo(tmp_path)
    root = tmp_path / "intake"
    assert main(["intake", f"file://{src}", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "next:    autoresearch intake propose" in out and "no box" in out
    draft = json.loads((root / "repo" / "draft.json").read_text())
    assert draft["package"] == "widget" and draft["run_id"] == "repo_w4"
    assert "## Benchmark sources" in (root / "repo" / "brief.md").read_text()


def test_intake_refuses_a_compiled_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _git_repo(tmp_path)
    (src / "widget" / "fast.pyx").write_text("cdef int x\n")
    for args in (["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"]):
        subprocess.run(["git", *args], cwd=src, check=True, capture_output=True)
    root = tmp_path / "intake"
    assert main(["intake", f"file://{src}", "--root", str(root)]) == 2
    assert "not in scope" in capsys.readouterr().out
    assert not (root / "repo" / "draft.json").exists()


def test_intake_propose_writes_the_config_and_refuses_to_overwrite_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autoresearch.model import sail_model

    src = _git_repo(tmp_path)
    root = tmp_path / "intake"
    assert main(["intake", f"file://{src}", "--root", str(root)]) == 0
    models: list[FakeChatModel] = [FakeChatModel([tool_call("submit_proposal", GOOD)])]
    monkeypatch.setattr(sail_model, "SailChatModel", lambda worker: models.pop(0))
    configs = tmp_path / "configs"
    args = ["intake", "propose", str(root / "repo"), "--configs", str(configs)]
    capsys.readouterr()
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "next:    autoresearch check" in out and "inputs: dense, sparse" in out
    cfg = load_config(configs / "repo_w4.toml")
    assert cfg.target.call == "w.parse(text)" and cfg.target.uncalibrated == ("dense", "sparse")
    assert json.loads((root / "repo" / "proposal.json").read_text())["axis"] == GOOD["axis"]

    # A second proposal would overwrite it: refused before any model exists.
    assert main(args) == 2
    assert "No model was called" in capsys.readouterr().err


# ----- profile and calibrate ---------------------------------------------------------


def test_profile_writes_the_documents_and_sets_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _fake_boxes(monkeypatch)
    cfg_path = _config(tmp_path)
    docs_root = tmp_path / "docs"
    assert main(["profile", str(cfg_path), "--docs-root", str(docs_root)]) == 0
    assert factory.created[0].terminated
    written = sorted(p.name for p in (docs_root / "t1_w4d").iterdir())
    assert written == [
        "profile_er1000_005_callers.txt",
        "profile_er1000_005_flat.txt",
        "profile_inputs.txt",
    ]
    cfg = load_config(cfg_path)
    assert cfg.target.docs == tuple(
        str(docs_root / "t1_w4d" / n) for n in (written[2], written[1], written[0])
    )
    inputs_doc = (docs_root / "t1_w4d" / "profile_inputs.txt").read_text()
    assert "gn800" in inputs_doc and "80.0%" in inputs_doc


def test_calibrate_writes_every_floor_into_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = _fake_boxes(monkeypatch, speedup=1.01)
    cfg_path = _config(tmp_path, PILOT.read_text().replace(GN800_FLOOR, "", 1))
    assert load_config(cfg_path).target.uncalibrated == ("gn800",)
    out = tmp_path / "cal"
    assert main(["calibrate", str(cfg_path), "--rounds", "1", "--out", str(out)]) == 0
    assert factory.created[0].terminated
    cfg = load_config(cfg_path)
    assert cfg.target.uncalibrated == ()
    assert all(i.noise_floor is not None and i.noise_floor > 1.0 for i in cfg.target.inputs)
    assert "null pairs" in cfg_path.read_text()
    assert len(list(out.glob("*.jsonl"))) == 1
    assert "written into" in capsys.readouterr().out


def test_calibrate_does_not_overwrite_a_config_edited_while_the_box_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = _config(tmp_path, PILOT.read_text().replace(GN800_FLOOR, "", 1))

    def edit_meanwhile(box: FakeBox, role: str) -> None:
        cfg_path.write_text(cfg_path.read_text() + "\n# a hand edit\n")

    _fake_boxes(monkeypatch, prepare=edit_meanwhile)
    assert main(["calibrate", str(cfg_path), "--rounds", "1", "--out", str(tmp_path / "c")]) == 1
    assert "changed while the box ran" in capsys.readouterr().err
    assert cfg_path.read_text().endswith("# a hand edit\n")
    assert load_config(cfg_path).target.uncalibrated == ("gn800",)


# ----- next --------------------------------------------------------------------------


def _check_record(runs: Path, config: Path, ts: str, failed: bool = False) -> None:
    cfg = load_config(config)
    (runs / "check").mkdir(parents=True, exist_ok=True)
    verdicts = [{"rule": "gn800 call length", "level": "fail" if failed else "pass", "detail": ""}]
    record = {
        "sha": cfg.target.sha,
        "target_hash": admissibility.admission_hash(cfg.target),
        "verdicts": verdicts,
    }
    (runs / "check" / f"{ts}.json").write_text(json.dumps(record))


def _next(config: Path, runs: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    rc = main(["next", str(config), "--out", str(runs), "--repo-root", str(ROOT)])
    return rc, capsys.readouterr().out


def test_next_walks_the_stages_in_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    bare = set_docs(PILOT.read_text().replace(GN800_FLOOR, "", 1), [])
    config = _config(tmp_path, bare)

    rc, out = _next(config, runs, capsys)
    assert rc == 0 and "not checked" in out and "autoresearch check" in out
    assert "1 referee box, size m, no model call" in out

    _check_record(runs, config, "20260912-100000")
    rc, out = _next(config, runs, capsys)
    assert rc == 0 and "autoresearch profile" in out

    _check_record(runs, config, "20260912-110000", failed=True)
    rc, out = _next(config, runs, capsys)
    assert rc == 1 and "check failed" in out and "gn800 call length" in out
    (runs / "check" / "20260912-110000.json").unlink()

    config.write_text(PILOT.read_text().replace(GN800_FLOOR, "", 1))
    rc, out = _next(config, runs, capsys)
    assert rc == 0 and "no floor for gn800" in out and "autoresearch calibrate" in out

    config.write_text(PILOT.read_text())
    rc, out = _next(config, runs, capsys)
    assert rc == 0 and "ready to run" in out and "--until 1" in out


def test_writing_floors_or_docs_keeps_a_check_current_and_a_setup_change_does_not() -> None:
    text = PILOT.read_text()
    base = admissibility.admission_hash(parse_config(text).target)
    assert admissibility.admission_hash(parse_config(set_docs(text, [])).target) == base
    no_floor = text.replace(GN800_FLOOR, "", 1)
    assert admissibility.admission_hash(parse_config(no_floor).target) == base
    other = text.replace("nx.gn_graph(800, seed=3)", "nx.gn_graph(801, seed=3)", 1)
    assert admissibility.admission_hash(parse_config(other).target) != base
