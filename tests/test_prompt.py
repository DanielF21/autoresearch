"""The worker prompt: the history index as a survey of mechanisms."""

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


def test_every_version_shares_the_mechanics_and_names_no_budget() -> None:
    assert sorted(prompt.VERSIONS) == ["v1", "v2", "v3", "v4"]
    for version in prompt.VERSIONS:
        text = prompt.system_prompt(version, "3.12.4")
        assert "Python 3.12.4" in text and "{python}" not in text
        for word in ("80 turns", "3600", "max_turns", "budget"):
            assert word not in text
        # The blocks no version may drift on.
        assert "current turn is your last" in text
        assert "made before your first measurement" in text
        assert "Measure to decide, not to explore" in text
        assert "line 1     one sentence naming the mechanism" in text
        assert "Choice     " in text and "Untried    " in text


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
