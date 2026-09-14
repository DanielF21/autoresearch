"""Intake: scope rules, derivation, the proposal loop and the config it writes.

Every repository here is a directory written by the test. Nothing is fetched
from the network and no model is called: the conversation is a FakeChatModel.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from autoresearch.config import load_config, parse_config
from autoresearch.intake import derive, propose, render, scope
from autoresearch.model.fake_model import FakeChatModel, malformed_call, text, tool_call

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "configs" / "t1_w4d.toml"

PYPROJECT = """
[project]
name = "Widget"
requires-python = ">=3.10"
dependencies = ["attrs>=23", "numpy; python_version > '3'"]

[project.optional-dependencies]
test = ["pytest>=8", "hypothesis"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

HOT = "def parse(text):\n    return [w for w in text.split() if w]\n"


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return root


def _widget(tmp_path: Path, layout: str = ".", **extra: str) -> Path:
    pkg = "widget" if layout == "." else f"{layout}/widget"
    files = {
        "pyproject.toml": PYPROJECT,
        f"{pkg}/__init__.py": "from .core import parse\n",
        f"{pkg}/core.py": HOT,
        "tests/test_core.py": "def test_parse():\n    pass\n",
        "docs/index.md": "docs\n",
        "benchmarks/bench_core.py": "import widget\n",
    }
    files.update(extra)
    return _write(tmp_path / "repo", files)


def _derive(repo: Path, url: str = "https://github.com/org/Widget.git", package: str = ""):  # type: ignore[no-untyped-def]
    return derive.derive(url, repo, "a" * 40, str(TEMPLATE), 4, package)


def _refusals(findings: list[scope.Finding]) -> list[str]:
    return [f.rule for f in findings if f.refused]


# ----- scope and derivation ----------------------------------------------------------


def test_a_flat_pure_python_repository_is_drafted(tmp_path: Path) -> None:
    draft, findings = _derive(_widget(tmp_path))
    assert draft is not None, findings
    assert (draft.name, draft.package, draft.package_root) == ("widget", "widget", ".")
    assert draft.pip == ("attrs", "numpy", "hypothesis")
    assert draft.tests_full == "tests"
    assert draft.allow == ("widget/**",)
    assert draft.deny == ("tests/**", "docs/**", "benchmarks/**")
    assert draft.run_id == "widget_w4"
    assert "benchmarks/bench_core.py" in draft.benchmarks
    warned = {f.rule for f in findings}
    assert {"hot path", "python version"} <= warned


def test_a_src_layout_is_imported_from_src(tmp_path: Path) -> None:
    draft, _ = _derive(_widget(tmp_path, "src"))
    assert draft is not None
    assert draft.package_root == "src" and draft.allow == ("src/widget/**",)
    assert draft.package_dir == "src/widget"


def test_tests_inside_the_package_are_the_suite_and_are_denied(tmp_path: Path) -> None:
    repo = _write(
        tmp_path / "repo",
        {
            "setup.cfg": "[metadata]\nname = widget\n",
            "widget/__init__.py": "",
            "widget/algo/core.py": HOT,
            "widget/algo/tests/test_core.py": "def test_x():\n    pass\n",
        },
    )
    draft, findings = _derive(repo)
    assert draft is not None, findings
    assert draft.tests_full == "widget"
    assert draft.deny[0] == "widget/**/tests/**"


@pytest.mark.parametrize(
    ("extra", "detail"),
    [
        ({"widget/_speed.pyx": "cdef int x\n"}, "compiled sources"),
        ({"widget/_speed.c": "int x;\n"}, "compiled sources"),
        ({"Cargo.toml": "[package]\n"}, "Rust"),
        (
            {"setup.py": "from setuptools import setup, Extension\nsetup(ext_modules=[])\n"},
            "extension modules",
        ),
        (
            {
                "pyproject.toml": PYPROJECT.replace(
                    'build-backend = "hatchling.build"', 'build-backend = "maturin"'
                )
            },
            "maturin",
        ),
        (
            {
                "pyproject.toml": PYPROJECT.replace(
                    'requires = ["hatchling"]', 'requires = ["Cython>=3"]'
                )
            },
            "cython",
        ),
    ],
)
def test_a_compiled_build_is_refused(tmp_path: Path, extra: dict[str, str], detail: str) -> None:
    draft, findings = _derive(_widget(tmp_path, **extra))
    assert draft is None
    refused = [f for f in findings if f.refused]
    assert refused and all(f.rule == "compiled build" for f in refused)
    assert any(detail in f.detail for f in refused)


def test_no_python_project_is_refused(tmp_path: Path) -> None:
    repo = _write(tmp_path / "repo", {"widget/__init__.py": "", "tests/test_x.py": ""})
    draft, findings = _derive(repo)
    assert draft is None and "python project" in _refusals(findings)


def test_no_tests_is_refused(tmp_path: Path) -> None:
    repo = _widget(tmp_path)
    (repo / "tests" / "test_core.py").unlink()
    (repo / "tests").rmdir()
    draft, findings = _derive(repo)
    assert draft is None and _refusals(findings) == ["tests"]


def test_several_packages_need_a_name_unless_one_matches(tmp_path: Path) -> None:
    repo = _widget(tmp_path, **{"gadget/__init__.py": "", "other/__init__.py": ""})
    draft, _ = _derive(repo)
    assert draft is not None and draft.package == "widget"  # matches the project name

    repo = _widget(tmp_path / "two", **{"gadget/__init__.py": ""})
    (repo / "widget" / "__init__.py").rename(repo / "widget" / "init.py")
    (repo / "other").mkdir()
    (repo / "other" / "__init__.py").write_text("")
    draft, findings = _derive(repo, url="https://github.com/org/nothing")
    assert draft is None and "package" in _refusals(findings)
    draft, _ = _derive(repo, url="https://github.com/org/nothing", package="gadget")
    assert draft is not None and draft.package == "gadget"


FOO_PYPROJECT = '[project]\nname = "foo"\n'


def _module_repo(tmp_path: Path, **extra: str) -> Path:
    files = {
        "pyproject.toml": FOO_PYPROJECT,
        "foo.py": HOT,
        "tests/test_foo.py": "def test_parse():\n    pass\n",
    }
    files.update(extra)
    return _write(tmp_path / "repo", files)


def test_a_single_module_named_after_the_project_is_drafted(tmp_path: Path) -> None:
    repo = _module_repo(tmp_path)
    draft, findings = _derive(repo, url="https://github.com/org/foo")
    assert draft is not None, findings
    assert (draft.package, draft.package_root, draft.single_module) == ("foo", ".", True)
    assert draft.allow == ("foo.py",) and draft.package_dir == "foo.py"
    assert draft.tests_full == "tests" and draft.deny == ("tests/**",)
    text = derive.brief(repo, draft)
    assert "single module `foo.py`, imported as `foo`" in text
    assert "- `foo.py` 2" in text and "Largest modules" not in text


def test_a_single_module_is_scanned_for_threads(tmp_path: Path) -> None:
    repo = _module_repo(tmp_path, **{"foo.py": "import threading\n" + HOT})
    _, findings = _derive(repo, url="https://github.com/org/foo")
    assert any(f.rule == "one core" and "foo.py" in f.detail for f in findings)


def test_a_package_directory_beats_a_stray_module_of_the_same_name(tmp_path: Path) -> None:
    repo = _module_repo(tmp_path, **{"foo/__init__.py": "", "foo/core.py": HOT})
    draft, findings = _derive(repo, url="https://github.com/org/foo")
    assert draft is not None, findings
    assert not draft.single_module and draft.allow == ("foo/**",)
    assert draft.package_dir == "foo"


def test_a_module_not_named_after_the_project_is_no_package(tmp_path: Path) -> None:
    repo = _write(
        tmp_path / "repo",
        {
            "pyproject.toml": '[project]\nname = "bar"\n',
            "setup.py": "from setuptools import setup\nsetup()\n",
            "tests/test_x.py": "",
        },
    )
    draft, findings = _derive(repo, url="https://github.com/org/bar")
    assert draft is None and _refusals(findings) == ["package"]
    assert "or a single module named after the project" in findings[0].detail
    assert scope.find_packages(repo, ("bar", "setup")) == [(".", "setup")]


def test_an_old_draft_without_single_module_loads_as_a_package(tmp_path: Path) -> None:
    draft, _ = _derive(_widget(tmp_path))
    assert draft is not None
    d = draft.to_dict()
    assert d["single_module"] is False
    del d["single_module"]
    assert derive.Draft.from_dict(d).single_module is False
    assert derive.Draft.from_dict(draft.to_dict()) == draft


def test_a_single_module_proposal_must_name_the_module(tmp_path: Path) -> None:
    repo = _module_repo(tmp_path, **{"other.py": HOT})
    draft, findings = _derive(repo, url="https://github.com/org/foo")
    assert draft is not None, findings
    template = load_config(TEMPLATE)
    good = dict(GOOD, hot_file="foo.py", tests_module="tests/test_foo.py")
    accepted = propose.validate(good, repo, draft, template, "now")
    assert not isinstance(accepted, str), accepted
    assert parse_config(accepted[1]).target.hot_file == "foo.py"
    rejected = propose.validate(dict(good, hot_file="other.py"), repo, draft, template, "now")
    assert isinstance(rejected, str) and "must be the single module, foo.py" in rejected


def test_threads_and_tox_are_warnings_not_refusals(tmp_path: Path) -> None:
    repo = _widget(tmp_path, **{"widget/pool.py": "import threading\n", "tox.ini": "[tox]\n"})
    draft, findings = _derive(repo)
    assert draft is not None
    rules = {f.rule: f.level for f in findings}
    assert rules["one core"] == "warn" and rules["test runner"] == "warn"


TOX = """[tox]
envlist = py312-{unit,doctest}

[testenv]
deps =
    unit: pytest
    unit: matplotlib; python_version < '3.15'
    doctest: sphinx
    -r requirements.txt
extras = unit: diagrams
"""


def test_tox_test_environments_add_their_deps_and_extras(tmp_path: Path) -> None:
    pyproject = PYPROJECT.replace(
        'test = ["pytest>=8", "hypothesis"]',
        'test = ["pytest>=8", "hypothesis"]\ndiagrams = ["railroad-diagrams"]\ndocs = ["furo"]',
    )
    repo = _widget(tmp_path, **{"tox.ini": TOX, "pyproject.toml": pyproject})
    draft, findings = _derive(repo)
    assert draft is not None, findings
    assert draft.pip == ("attrs", "numpy", "hypothesis", "railroad-diagrams", "matplotlib")
    text = derive.brief(repo, draft)
    assert (
        "- `docs`: furo" in text and "diagrams" not in text.split("not installed")[1].split("##")[0]
    )


def test_repo_name() -> None:
    assert scope.repo_name("https://github.com/pyparsing/pyparsing") == "pyparsing"
    assert scope.repo_name("https://github.com/org/Py-Thing.git/") == "py_thing"


def _git_repo(tmp_path: Path) -> Path:
    src = _widget(tmp_path / "origin")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(["git", *args], cwd=src, check=True, capture_output=True)
    return src


def test_clone_pins_head_and_refuses_to_clone_twice(tmp_path: Path) -> None:
    src = _git_repo(tmp_path)
    dest = tmp_path / "intake" / "widget" / "repo"
    sha = scope.clone(f"file://{src}", dest)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert sha == head and (dest / "widget" / "core.py").is_file()
    with pytest.raises(scope.IntakeError, match="already exists"):
        scope.clone(f"file://{src}", dest)
    with pytest.raises(scope.IntakeError, match="not a repository URL"):
        scope.clone("--upload-pack=touch x", tmp_path / "elsewhere")


# ----- the proposal ------------------------------------------------------------------

GOOD: dict[str, Any] = {
    "hot_file": "widget/core.py",
    "alias": "w",
    "call": "w.parse(text)",
    "tests_module": "tests/test_core.py",
    "extra_pip": ["regex"],
    "axis": "text length and word density",
    "axis_reason": "a split based fast path wins on dense text and loses on sparse text",
    "inputs": [
        {
            "name": "dense",
            "setup": "import random\nrnd = random.Random(1 + SEED)\ntext = ' '.join(str(rnd.random()) for _ in range(1000))\n",
            "regime": "dense, short words",
            "why": "the list comprehension dominates",
        },
        {
            "name": "sparse",
            "setup": "text = 'x' * (1 + SEED % 7) + ' ' * 100000 + '[[target.inputs]]'",
            "regime": "sparse",
            "why": "whitespace scanning dominates",
        },
    ],
}


def _draft_dir(tmp_path: Path) -> tuple[Path, derive.Draft]:
    out = tmp_path / "intake" / "widget"
    repo = _widget(out)
    repo.rename(out / derive.REPO_DIR)
    draft, findings = _derive(out / derive.REPO_DIR)
    assert draft is not None, findings
    derive.write_draft(out, out / derive.REPO_DIR, draft)
    return out, draft


def _propose(
    out: Path, draft: derive.Draft, model: FakeChatModel, max_turns: int = 10
) -> propose.Outcome:
    return propose.propose(
        model,
        out,
        draft,
        load_config(TEMPLATE),
        (out / derive.BRIEF_FILE).read_text(),
        when="2026-09-12 12:00:00",
        max_turns=max_turns,
        prompt="system",
    )


def test_repair_continues_the_saved_conversation_with_the_check_report(tmp_path: Path) -> None:
    out, draft = _draft_dir(tmp_path)
    _propose(out, draft, FakeChatModel([tool_call("submit_proposal", GOOD, "c1")]))
    first = json.loads((out / "messages.json").read_text())
    fixed = dict(GOOD, call="w.parse(text * 2)")
    model = FakeChatModel([tool_call("submit_proposal", fixed, "c2")])
    report = "FAIL  dense call length  0.0010s is below 5ms"
    outcome = propose.repair(
        model,
        out,
        draft,
        load_config(TEMPLATE),
        report,
        when="2026-09-13 12:00:00",
        max_turns=5,
    )
    sent = model.requests[0]
    assert sent[: len(first)] == first
    assert sent[-1]["role"] == "user" and report in sent[-1]["content"]
    assert parse_config(outcome.text).target.call == "w.parse(text * 2)"
    assert model.cache_keys == ["intake-widget"]
    assert json.loads((out / "usage.json").read_text())["prompt_tokens"] == 200
    assert len(json.loads((out / "messages.json").read_text())) == len(sent) + 2


def test_tools_read_the_clone_and_never_leave_it(tmp_path: Path) -> None:
    out, _ = _draft_dir(tmp_path)
    repo = out / derive.REPO_DIR
    (tmp_path / "secret.txt").write_text("key")
    assert "widget/core.py" in propose.list_files(repo, {"path": "widget"})
    assert "     1  def parse(text):" in propose.read_file(repo, {"path": "widget/core.py"})
    assert "widget/core.py:2:" in propose.grep(repo, {"pattern": r"split\(\)"})
    for bad in ("../../secret.txt", "/etc/passwd"):
        with pytest.raises(propose.ToolError, match="outside"):
            propose.read_file(repo, {"path": bad})
    (repo / "link.py").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(propose.ToolError, match="outside"):
        propose.read_file(repo, {"path": "link.py"})
    assert "key" not in propose.grep(repo, {"pattern": "key", "glob": "*"})


def test_a_rejected_proposal_gets_its_reasons_and_an_accepted_one_loads(tmp_path: Path) -> None:
    out, draft = _draft_dir(tmp_path)
    bad = dict(GOOD, call="w.parse(", hot_file="tests/test_core.py", inputs=GOOD["inputs"][:1])
    model = FakeChatModel(
        [
            tool_call("list_files", {"path": "."}, "c1"),
            tool_call("read_file", {"path": "../outside"}, "c2"),
            tool_call("submit_proposal", bad, "c3"),
            tool_call("submit_proposal", GOOD, "c4"),
        ]
    )
    outcome = _propose(out, draft, model)
    replies = {m["tool_call_id"]: m["content"] for m in model.requests[-1] if m["role"] == "tool"}
    assert replies["c2"].startswith("error:") and "outside" in replies["c2"]
    assert replies["c3"].startswith("not accepted:")
    for reason in ("inside the package", "not one expression", "2 to 12 inputs"):
        assert reason in replies["c3"]
    assert outcome.turns == 4
    # A setup that ignores SEED is one point in input space, and refused by name.
    point = dict(GOOD["inputs"][0], setup="text = 'a b c' * 1000")
    result = propose.validate(
        dict(GOOD, inputs=[point, GOOD["inputs"][1]]),
        out / derive.REPO_DIR,
        draft,
        load_config(TEMPLATE),
        "now",
    )
    assert isinstance(result, str) and "setup of dense does not read SEED" in result
    # A result built around the library's, with a seed tag beside it, is refused:
    # the first pycodestyle proposal under the seed rule did exactly that.
    for tagged in ("[w.parse(text), tag]", "(w.parse(text), SEED)", "w.parse(text) + [SEED]"):
        result = propose.validate(
            dict(GOOD, call=tagged), out / derive.REPO_DIR, draft, load_config(TEMPLATE), "now"
        )
        assert isinstance(result, str) and "call must be one function call" in result
    for plain in ("w.parse(text)", "list(w.stream(text))", "sorted(w.parse(text), key=len)"):
        result = propose.validate(
            dict(GOOD, call=plain), out / derive.REPO_DIR, draft, load_config(TEMPLATE), "now"
        )
        assert not isinstance(result, str)
    # Nothing in the text goes stale once profile and calibrate write docs and floors.
    for stale in ("Not admitted", "noise_floor is absent", "profile writes these"):
        assert stale not in outcome.text

    cfg = parse_config(outcome.text)
    t = cfg.target
    assert (t.package, t.alias, t.call, t.hot_file) == (
        "widget",
        "w",
        "w.parse(text)",
        "widget/core.py",
    )
    assert t.pip == ("attrs", "numpy", "hypothesis", "regex")
    assert t.uncalibrated == ("dense", "sparse") and t.docs == ()
    assert t.inputs[1].setup == GOOD["inputs"][1]["setup"] + "\n"
    template = load_config(TEMPLATE)
    assert cfg.worker == template.worker and cfg.referee == template.referee
    assert "input axis: text length and word density" in outcome.text
    saved = json.loads((out / "messages.json").read_text())
    assert saved[0] == {"role": "system", "content": "system"}
    assert json.loads((out / "usage.json").read_text())["prompt_tokens"] == 400


def test_a_cut_off_tool_call_is_refused_and_the_conversation_goes_on(tmp_path: Path) -> None:
    """pycodestyle's intake on 2026-09-14 ended at reply 38: the model cut a
    read_file call off mid JSON, the harness ran it with no path, and the endpoint
    refused every later request over the invalid arguments in the history."""
    out, draft = _draft_dir(tmp_path)
    cut = '{"path": "reading? Let\'s read testing/data/W60.py", "count": 80'
    model = FakeChatModel(
        [malformed_call("read_file", cut, "c1"), tool_call("submit_proposal", GOOD, "c2")]
    )
    outcome = _propose(out, draft, model)
    assert outcome.turns == 2
    replies = {m["tool_call_id"]: m["content"] for m in model.requests[-1] if m["role"] == "tool"}
    assert replies["c1"].startswith("error:") and "not valid JSON" in replies["c1"]
    assert "W60.py" in replies["c1"] and "is not a file" not in replies["c1"]
    echoed = next(m for m in model.requests[-1] if m["role"] == "assistant")
    assert echoed["tool_calls"][0]["function"]["arguments"] == "{}"


def test_the_turn_cap_stops_the_conversation_and_keeps_the_messages(tmp_path: Path) -> None:
    out, draft = _draft_dir(tmp_path)
    model = FakeChatModel([text("thinking") for _ in range(5)])
    with pytest.raises(propose.ProposeError, match="5 turns"):
        _propose(out, draft, model, max_turns=5)
    messages = json.loads((out / "messages.json").read_text())
    assert any("replies left" in str(m.get("content")) for m in messages)


def test_a_setup_that_cannot_be_a_literal_string_still_round_trips(tmp_path: Path) -> None:
    _, draft = _draft_dir(tmp_path)
    setup = "a = '''x'''\nb = \"\\t\\\\\"\n"
    proposal = render.Proposal(
        hot_file="widget/core.py",
        alias="w",
        call="w.parse(a)",
        tests_module="tests",
        extra_pip=(),
        axis="a",
        axis_reason="b",
        inputs=(
            render.ProposedInput("one", setup, "r", "w"),
            render.ProposedInput("two", "a = 1", "r", "w"),
        ),
    )
    cfg = parse_config(render.render(draft, proposal, load_config(TEMPLATE), "now"))
    assert cfg.target.inputs[0].setup == setup
    assert "fingerprint" not in render.render(draft, proposal, load_config(TEMPLATE), "now")
