"""The worker prompt: the history index as a survey of mechanisms."""

from pathlib import Path

from autoresearch.config import TargetSpec, load_config
from autoresearch.types import (
    Attempt,
    AttemptRef,
    InputTiming,
    Measurement,
    PairTiming,
    Prediction,
    StopReason,
    SuiteResult,
    Usage,
)
from autoresearch.worker import prompt
from autoresearch.worker.prompt.render import index_line, outcome
from tests.helpers import BASE_SHA


def _attempt(number: int, rationale: str, speedup: float | None = None) -> Attempt:
    measurement = None
    if speedup is not None:
        pair = PairTiming(0, "base_first", 0, 1.0, 1.0 / speedup, contaminated=False)
        measurement = Measurement(
            applied=True,
            tests=(
                SuiteResult("module", 1, 0, 0, 1.0, True),
                SuiteResult("full", 1, 0, 0, 1.0, True),
            ),
            inputs=(InputTiming("dense", 1.01, "a", "a", (pair,), speedup),),
        )
    return Attempt(
        ref=AttemptRef(number, 1, 0),
        base_sha=BASE_SHA,
        patch="diff" if speedup is not None else None,
        prediction=Prediction(1.2),
        rationale=rationale,
        stop_reason=StopReason.SUBMITTED if speedup is not None else StopReason.MAX_TURNS,
        usage=Usage(),
        wall_s=1.0,
        measurement=measurement,
    )


def test_the_mechanism_line_is_the_first_non_empty_line_cut_short() -> None:
    assert prompt.mechanism_line("\n\n# Bitmask the neighbour sets.\nMore.") == (
        "Bitmask the neighbour sets."
    )
    assert prompt.mechanism_line("") == "(no rationale)"
    long = "x" * 100
    cut = prompt.mechanism_line(long)
    assert len(cut) == prompt.SUMMARY_CHARS and cut.endswith("…")


def test_the_index_carries_each_rationale_first_line() -> None:
    history = (
        _attempt(1, "Bitmask the neighbour sets.\n\nChoice: depart.", 3.0),
        _attempt(2, "", None),
        _attempt(3, "Cache degrees per node.\nChoice: extend, on 0001.", 2.0),
    )
    text = prompt.render_history(history)
    lines = text.splitlines()
    header = next(ln for ln in lines if ln.startswith("id "))
    assert header.split()[-1] == "mechanism"
    rows = {ln.split()[0]: ln for ln in lines if ln[:4].isdigit()}
    assert rows["0001"].endswith("Bitmask the neighbour sets.")
    assert "3.00x" in rows["0001"] and "real speedup" in rows["0001"]
    assert rows["0002"].endswith("(no rationale)") and "not measured yet" in rows["0002"]
    assert rows["0003"].endswith("Cache degrees per node.")
    assert "best 3.00x" in text and "Survey the whole table" in text
    # A rationale's later lines never reach the index.
    assert "Choice: extend" not in text


def test_the_outcome_names_a_patch_that_recognised_its_input() -> None:
    """pycodestyle_p3_w16 0233 would have read as 'real speedup 2985x'. The
    index says what the referee established instead, and the row's geomean is
    the held out score."""
    from dataclasses import replace

    a = _attempt(4, "Match the six benchmark files by string equality.", 1.0)
    assert a.measurement is not None
    fitted = replace(a.measurement.inputs[0], shown_speedup=2985.0, overfit=True)
    a = replace(a, measurement=replace(a.measurement, inputs=(fitted,)))
    assert outcome(a) == "overfit on dense"
    row = index_line(a)
    assert "1.00x" in row and "2985" not in row and "overfit on dense" in row


def test_the_target_section_explains_the_seed_and_the_held_out_instance() -> None:
    text = prompt.render_target(_target(), BASE_SHA, 6)
    assert "`SEED` to an integer" in text
    assert "one seed you are never shown" in text and "recorded as overfit" in text
    assert "walk over its result" in text
    assert "seed=3 + SEED" in text


def test_every_version_shares_the_mechanics_and_names_no_budget() -> None:
    assert sorted(prompt.VERSIONS) == ["v0", "v1", "v2", "v3", "v4"]
    for version in prompt.VERSIONS:
        text = prompt.system_prompt(version, "3.12.4")
        assert "Python 3.12.4" in text and "{python}" not in text and "{index_note}" not in text
        for word in ("80 turns", "3600", "max_turns", "budget"):
            assert word not in text
        if version == "v0":
            continue  # the frozen width experiment prompt predates these blocks
        # The blocks no version may drift on.
        assert "current turn is your last" in text
        assert "made before your first measurement" in text
        assert "Measure to decide, not to explore" in text
        assert "line 1     one sentence naming the mechanism" in text
        assert "Choice     " in text and "Untried    " in text


def _target() -> TargetSpec:
    return load_config(Path(__file__).parent.parent / "configs" / "t1_w4d.toml").target


def test_v0_is_the_width_experiment_prompt_with_its_own_closing() -> None:
    text = prompt.system_prompt("v0", "3.12.4")
    assert text.startswith(
        "You are a performance engineer working alone in a sandbox on one repository."
    )
    assert "The history table below is only an index of these." in text
    assert "current turn is your last" not in text
    # The sandbox note argument is for the composed versions; v0 ignores it.
    assert prompt.system_prompt("v0", "3.12.4", history_index=False) == text
    old = prompt.initial_user_message(
        _target(), BASE_SHA, (), (), 7, 6, closing=prompt.closing("v0")
    )
    assert old.endswith(
        "You are attempt 0007. Begin by reading the hot file and the history with the "
        "shell, then make one change and submit."
    )
    new = prompt.initial_user_message(_target(), BASE_SHA, (), (), 7, 6)
    assert new.endswith("write down what you expect before you measure, then make your change.")
    assert prompt.closing("v1") == prompt.CLOSING


def test_the_sandbox_note_follows_history_index() -> None:
    with_index = prompt.system_prompt("v1", "3.12.4")
    assert "The history table in the first message is an\n" in with_index
    assert "index of these." in with_index
    without = prompt.system_prompt("v1", "3.12.4", history_index=False)
    assert "there is no index, so the directories are the" in without
    assert "history table" not in without
    # The default output is what the four versions were written with: the note
    # sits inside the indented filesystem block, wrapped to it.
    assert "                       index of these.\n" in with_index


def test_hidden_attempts_are_announced_in_every_shape_of_the_section() -> None:
    history = (_attempt(1, "Bitmask the neighbour sets.", 3.0),)
    assert "withheld" not in prompt.render_history(history)
    told = prompt.render_history(history, hidden=2)
    assert "2 attempts from the record are withheld from you this round" in told
    assert "numbering has gaps" in told and "Bitmask" in told
    one = prompt.render_history(history, hidden=1)
    assert "1 attempt from the record is withheld" in one
    # Everything withheld: the slot is not told it is first.
    empty = prompt.render_history((), hidden=3)
    assert "You are first" not in empty
    assert "3 attempts from the record are withheld" in empty
    assert "/workspace/history does not exist for you" in empty


def test_no_index_rendering_names_the_directories_and_no_scores() -> None:
    history = (
        _attempt(1, "Bitmask the neighbour sets.", 3.0),
        _attempt(2, "", None),
        _attempt(3, "Cache degrees per node.", 2.0),
    )
    text = prompt.initial_user_message(_target(), BASE_SHA, (), history, 4, 6, index=False)
    assert "3 earlier attempts" in text
    assert "measurement.json holds the geomean" in text
    assert "rationale.md opens with the line naming the mechanism" in text
    for absent in (
        "best 3.00x",
        "3.00x",
        "real speedup",
        "Bitmask",
        "Cache degrees",
        "mechanism\n",
    ):
        assert absent not in text
    assert "withheld" not in text
    told = prompt.render_history(history, hidden=2, index=False)
    assert "2 attempts from the record are withheld" in told and "3.00x" not in told


def test_each_version_has_its_own_job() -> None:
    v1, v2, v3, v4 = (prompt.system_prompt(v, "3") for v in ("v1", "v2", "v3", "v4"))
    assert "choose one of three things" in v1 and "survey it" in v1
    assert "Your job in particular is to depart" in v2 and "Read nothing else" in v2
    assert "Your job in particular is to combine" in v3 and "the best has not absorbed" in v3
    assert "Your job in particular is to shrink" in v4 and "90 percent" in v4
    # Each says what to do when it is first, since round 1 has no history.
    for text in (v2, v3, v4):
        assert "you are first" in text
    assert len({v1, v2, v3, v4}) == 4
    assert prompt.DEFAULT_VERSION == "v1"
