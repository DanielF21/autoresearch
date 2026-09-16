"""What AlphaEvolve's model is told, and that the harness worker's text did not change."""

from __future__ import annotations

from alphaevolve import prompt
from alphaevolve.blocks import MARK_END, MARK_START, derive
from alphaevolve.meta import MetaPrompt
from autoresearch.config import load_config
from autoresearch.worker.prompt import render_target
from tests.ae_helpers import FLAT, HOT, ROOT, SEARCH, SOURCE
from tests.helpers import BASE_SHA, TEST_CONFIG

TARGET = load_config(TEST_CONFIG).target
BLOCKS = derive(FLAT, {HOT: SOURCE}, TARGET.allow, TARGET.deny, 10_000)


def test_the_harness_target_section_is_byte_identical() -> None:
    snapshot = (ROOT / "tests" / "fixtures" / "render_target_t1_w4d.txt").read_text()
    assert render_target(TARGET, BASE_SHA, 6) == snapshot


def test_without_tools_the_target_section_says_nothing_about_run_benchmark() -> None:
    text = render_target(TARGET, BASE_SHA, 6, tools=False, editable=(HOT,))
    assert "run_benchmark" not in text and "Allowed files" not in text
    assert f"Editable: only the marked blocks in {HOT}." in text
    assert "You are shown seed 0. The referee times seed 0" in text
    assert text.endswith("a guard on the wrong property will not prevent that.\n")


def test_the_system_prompt_carries_the_referee_rules_and_no_sandbox() -> None:
    text = prompt.system_prompt()
    for start in prompt.SHARED_BULLETS:
        assert f"- {start}" in text
    for word in ("git apply", "/workspace", "shell", "run_tests", "your last"):
        assert word not in text


def _prefix() -> str:
    return prompt.constant_prefix(
        TARGET, BASE_SHA, 6, (("profile.txt", "rows"),), {HOT: SOURCE}, BLOCKS
    )


def test_the_prefix_is_constant_and_shows_the_marked_source() -> None:
    assert _prefix() == _prefix()
    text = _prefix()
    assert f"{MARK_START}\ndef clustering(G):" in text and f"    return total\n{MARK_END}" in text
    assert "### profile.txt" in text
    assert "run_benchmark" not in text and "/workspace" not in text


def test_the_variable_part_follows_openevolves_sections() -> None:
    base = prompt.Shown("base", 1.0, tuple(b.text for b in BLOCKS), None)
    text = prompt.variable_part("", base, [base], [base], [], [base], BLOCKS)
    for heading in (
        "## Instructions evolved in this run\n\n(none yet)",
        "# Current Program Information\n- Fitness: 1.0000",
        "## Previous Attempts",
        "## Top Performing Programs",
        "## Inspiration Programs",
        "# Current Program\n",
        "# Task\n",
        SEARCH,
    ):
        assert heading in text
    assert "## Diverse Programs" not in text
    assert f"#### {HOT}, lines 4 to 8 at the base commit: clustering" in text


def test_the_meta_task_lists_the_population() -> None:
    text = prompt.meta_part(
        [MetaPrompt("m0000", "", 2, 1.3), MetaPrompt("m0005", "Use sets.", 0, None)]
    )
    assert "### m0000: used by 2 candidates, best fitness 1.3000" in text
    assert "### m0005: used by 0 candidates, best fitness none yet\nUse sets." in text
    assert "<instructions>" in text
