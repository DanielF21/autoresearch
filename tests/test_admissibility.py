"""The rules a target is admitted by, judged from a survey of the base tree."""

from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.config import RunConfig, SuitePaths, load_config
from autoresearch.referee import admissibility as adm
from autoresearch.referee.referee import LAUNCH_TIMEOUT, TESTS_TIMEOUT, InputSurvey, Referee, Survey
from autoresearch.types import SuiteResult
from tests.helpers import referee_box

ROOT = Path(__file__).parent.parent


@pytest.fixture
def config() -> RunConfig:
    return load_config(ROOT / "configs" / "t1_w4d.toml")


def _input(name: str, **kw: object) -> InputSurvey:
    base = dict(
        hot_executed=True,
        hot_share=0.9,
        call_s=0.5,
        import_s=0.1,
        setup_s=0.05,
        fixed_s=0.3,
        python="3.12.4",
        fingerprints=("abc", "abc"),
    )
    base.update(kw)
    return InputSurvey(name=name, **base)  # type: ignore[arg-type]


def _suite(scope: str, ok: bool = True, duration_s: float = 60.0) -> SuiteResult:
    return SuiteResult(scope, 10, 0 if ok else 1, 0, duration_s, ok)


def _good() -> Survey:
    return Survey(inputs=(_input("a"), _input("b")), tests=(_suite("module"), _suite("full")))


def _levels(verdicts: tuple[adm.Verdict, ...]) -> dict[str, str]:
    return {v.rule: v.level for v in verdicts}


def test_a_target_that_meets_every_rule_is_admissible(config: RunConfig) -> None:
    verdicts = adm.judge(config.target, _good(), 7)
    assert not any(v.failed for v in verdicts)
    assert "admissible" in adm.render(config.target, _good(), verdicts).splitlines()[-1]


def test_an_input_that_cannot_run_fails_by_name(config: RunConfig) -> None:
    survey = replace(_good(), inputs=(_input("a"), InputSurvey("b", error="ImportError: x")))
    levels = _levels(adm.judge(config.target, survey, 7))
    assert levels["a runs"] == "pass" and levels["b runs"] == "fail"
    assert "b hot file executes" not in levels  # nothing else is judged for it


def test_a_hot_file_that_did_not_execute_fails(config: RunConfig) -> None:
    survey = replace(_good(), inputs=(_input("a", hot_executed=False),))
    verdicts = adm.judge(config.target, survey, 7)
    assert _levels(verdicts)["a hot file executes"] == "fail"
    assert config.target.hot_file in next(v.detail for v in verdicts if v.failed)


def test_a_small_hot_share_warns_but_does_not_fail(config: RunConfig) -> None:
    survey = replace(_good(), inputs=(_input("a", hot_share=0.1),))
    verdicts = adm.judge(config.target, survey, 7)
    assert _levels(verdicts)["a hot file executes"] == "warn"
    assert not any(v.failed for v in verdicts)


def test_a_result_that_differs_between_two_launches_fails(config: RunConfig) -> None:
    survey = replace(_good(), inputs=(_input("a", fingerprints=("abc", "abd")),))
    verdicts = adm.judge(config.target, survey, 7)
    assert _levels(verdicts)["a deterministic"] == "fail"


def test_the_call_must_sit_inside_the_band_one_launch_can_time(config: RunConfig) -> None:
    limit = adm.call_max_s(7)
    assert limit == (LAUNCH_TIMEOUT - adm.LAUNCH_HEADROOM_S) / 7
    short = replace(_good(), inputs=(_input("a", call_s=0.001),))
    long = replace(_good(), inputs=(_input("a", call_s=limit + 1),))
    edge = replace(_good(), inputs=(_input("a", call_s=limit - 1),))
    assert _levels(adm.judge(config.target, short, 7))["a call length"] == "fail"
    assert _levels(adm.judge(config.target, long, 7))["a call length"] == "fail"
    assert _levels(adm.judge(config.target, edge, 7))["a call length"] == "pass"
    # Fewer repeats per launch leave room for a longer call.
    assert _levels(adm.judge(config.target, long, 3))["a call length"] == "pass"


def test_both_suites_must_pass_and_fit_the_timeout(config: RunConfig) -> None:
    failing = replace(_good(), tests=(_suite("module"), _suite("full", ok=False)))
    assert _levels(adm.judge(config.target, failing, 7))["full suite"] == "fail"
    slow = replace(_good(), tests=(_suite("module"), _suite("full", duration_s=TESTS_TIMEOUT)))
    assert _levels(adm.judge(config.target, slow, 7))["full suite"] == "fail"
    missing = replace(_good(), tests=(_suite("module"),), errors=("full tests: guest died",))
    verdicts = adm.judge(config.target, missing, 7)
    assert _levels(verdicts)["full suite"] == "fail"
    assert "guest died" in next(v.detail for v in verdicts if v.rule == "full suite")


def test_one_suite_named_twice_warns_and_uncalibrated_inputs_are_noted(
    config: RunConfig,
) -> None:
    same = replace(config.target, tests=SuitePaths("tests", "tests"))
    assert _levels(adm.judge(same, _good(), 7))["two suites"] == "warn"
    first, *rest = config.target.inputs
    bare = replace(config.target, inputs=(replace(first, noise_floor=None), *rest))
    verdicts = adm.judge(bare, _good(), 7)
    assert _levels(verdicts)["noise floors"] == "note"
    assert first.name in next(v.detail for v in verdicts if v.rule == "noise floors")
    assert not any(v.failed for v in verdicts)


def test_render_names_every_failed_rule_and_the_verdict(config: RunConfig) -> None:
    survey = replace(_good(), inputs=(_input("a", hot_executed=False),))
    verdicts = adm.judge(config.target, survey, 7)
    text = adm.render(config.target, survey, verdicts)
    assert "FAIL  a hot file executes" in text
    assert text.splitlines()[-1].startswith("not admissible: 1 rule(s) failed")
    assert "python 3.12.4 in the box" in text
    assert "module suite: ok" in text


def test_the_referee_surveys_the_base_tree_twice_per_input_and_both_suites(
    config: RunConfig,
) -> None:
    """No patch, no floors needed, and the worktree is cleaned up after."""
    first, *rest = config.target.inputs
    bare = replace(
        config, target=replace(config.target, inputs=(replace(first, noise_floor=None), *rest))
    )
    box = referee_box()
    ref = Referee(box, bare)
    ref.setup()
    survey = ref.survey()
    n = len(config.target.inputs)
    verifies = [c for c in box.commands if "--verify" in c]
    assert len(verifies) == 2 * n and all("/base " in c or c.endswith("/base") for c in verifies)
    assert not any("--patch" in c or "--repeats" in c or "canary.py" in c for c in box.commands)
    assert [i.name for i in survey.inputs] == [i.name for i in config.target.inputs]
    assert all(i.deterministic and i.hot_executed and i.python == "3.12.4" for i in survey.inputs)
    assert [t.scope for t in survey.tests] == ["module", "full"]
    assert sum(1 for c in box.commands if "--remove" in c) == 2
    assert ref.broken == ""
