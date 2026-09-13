"""The deterministic gates on a draft, the dossier, and judge normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.model.fake_model import FakeChatModel, tool_call
from autoresearch.scribe import checks, dossier, runread
from autoresearch.scribe import facts as facts_mod
from autoresearch.scribe import method as method_mod
from autoresearch.scribe.corpus_stats import corpus_stats
from autoresearch.scribe.judges import common
from autoresearch.scribe.judges.discrimination import discriminate
from autoresearch.scribe.judges.novelty import judge_novelty
from autoresearch.scribe.loop import Caps
from tests.scribe_helpers import make_run, pr

PATCH = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n-LIMIT = 64\n+LIMIT = 4096\n x = 1\n"
)
CAPS = Caps(3, 600, 1_000_000)
V1 = method_mod.load_method(method_mod.PACKAGED / "v1")


def candidate(tmp_path: Path) -> tuple[runread.Candidate, facts_mod.Facts]:
    run = make_run(
        tmp_path,
        [
            {"patch": PATCH, "speedups": {"dense": 6.31, "sparse": 2.61}},
            {
                "patch": PATCH.replace("4096", "9"),
                "speedups": {"dense": 3.0, "sparse": 0.5},
                "rationale": "Tried a smaller table.\n\nMore.",
            },
        ],
    )
    first, _ = runread.load_candidates(run)
    return first, facts_mod.compute_facts(PATCH)


def sheet_for(tmp_path: Path) -> checks.FactSheet:
    c, f = candidate(tmp_path)
    return checks.build_fact_sheet(c, f)


@pytest.mark.parametrize(
    "text",
    [
        "Clustering is 6.31x faster on dense graphs and 2.61x on sparse ones.",
        "About 6.3x faster overall.",  # rounded to the digits shown
        "The table grows from 64 to 4096 entries.",  # diff literals
        "The full suite passes: 9090 passed, 0 failed.",  # test counts, with a thousands form below
        "9,090 tests pass.",
        "That is 531% faster.",  # (6.31 - 1) * 100
        "It takes 84% less time.",  # (1 - 1/6.31) * 100
        "Each input ran 6 pairs; median base time 1.000 s.",
        "1. first point\n2. second point",  # list markers are not numbers
        "See [the benchmark](https://example.com/run/123456).",  # link targets are dropped
        "Setup: make(100, 0.5) and make(100, 0.01).",
    ],
)
def test_numbers_that_trace_to_the_record_pass(tmp_path: Path, text: str) -> None:
    result = checks.trace_numbers("ENH: faster table", text, sheet_for(tmp_path))
    assert result.ok, result.detail


@pytest.mark.parametrize(
    ("text", "stray"),
    [
        ("Clustering is 7.2x faster.", "7.2x"),
        ("It is 6.4x faster.", "6.4x"),  # 6.31 does not round to 6.4
        ("Closes #8812.", "#8812"),
        ("Tested on Python 3.12.", "3.12"),
        ("Median base time 250 ms.", "250 ms"),
    ],
)
def test_numbers_that_the_record_does_not_hold_fail_by_name(
    tmp_path: Path, text: str, stray: str
) -> None:
    result = checks.trace_numbers("ENH: x", text, sheet_for(tmp_path))
    assert not result.ok
    assert any(d.startswith(stray) for d in result.detail), result.detail


def test_identifiers_and_hashes_are_not_numbers() -> None:
    tokens = checks.extract_numbers("er1000_005 at c94928ed and 0e75965, O(n^3), x2")
    assert tokens == []


def test_direction_must_agree_with_the_measurement(tmp_path: Path) -> None:
    sheet = sheet_for(tmp_path)
    assert checks.check_direction("t", "6.31x faster", sheet).ok
    bad = checks.check_direction("t", "now 6.31x slower than before", sheet)
    assert not bad.ok and "called slower" in bad.detail[0]


def test_coverage_accepts_a_name_or_the_setup_expression(tmp_path: Path) -> None:
    sheet = sheet_for(tmp_path)
    assert checks.check_coverage("dense graphs: make(100, 0.5); sparse too", sheet).ok
    missing = checks.check_coverage("only make(100,0.5) mentioned", sheet)
    assert not missing.ok and missing.detail == ("sparse is not mentioned by name or setup",)


def test_lint_flags_slop_but_drops_patterns_the_corpus_itself_uses() -> None:
    body = "This change leverages a bitset. Furthermore, it is significantly faster — really."
    result, dropped = checks.lint("ENH: x", body, V1, ["A plain body.", "Another plain body."])
    assert not result.ok and dropped == ()
    names = " ".join(result.detail)
    assert "buzzwords" in names and "connectives" in names and "em dash" in names

    em_dash_corpus = [f"Body {i} — with a dash." for i in range(5)]
    _, dropped = checks.lint("ENH: x", body, V1, em_dash_corpus)
    assert "em dash" in dropped


def test_structure_is_judged_against_the_corpus_ranges() -> None:
    stats = corpus_stats(
        [
            pr(1, "ENH: speed up the thing", "x" * 300),
            pr(2, "BUG: fix the other thing", "y" * 900 + "\n\n- one\n- two"),
        ]
    )
    assert checks.check_structure("ENH: faster thing", "z" * 500, stats).ok
    bad = checks.check_structure(
        "PERF: faster thing", "## Summary\n\n" + "z" * 50 + "\n<!-- left over -->", stats
    )
    text = " ".join(bad.detail)
    assert not bad.ok
    assert "title prefix 'PERF'" in text and "headers" in text and "characters" in text
    assert "template comments" in text


def test_the_dossier_carries_the_search_log_and_never_the_run_config(tmp_path: Path) -> None:
    c, f = candidate(tmp_path)
    run = tmp_path / "fake_run"
    target = runread.read_target(run)
    (_, other) = runread.load_candidates(run)
    text = dossier.build_dossier(
        c, f, target, [(other, facts_mod.compute_facts(other.patch or ""))], "<!-- tpl -->"
    )
    assert "attempt 2 [same file]: slower on sparse 0.50x" in text
    assert "notes: Tried a smaller table." in text and "More." not in text
    assert "| dense | `G = make(100, 0.5)` | 6.31x |" in text
    assert "module suite: 56 passed" in text and target.sha in text
    assert "crossover" not in text
    assert dossier.rotate(tuple(pr(i, "t", "b") for i in range(5)), 3, 4)[0].number == 4


def test_anonymise_treats_real_and_generated_bodies_alike() -> None:
    raw = (
        "Fixes #123 by @alice, see [bench](https://x.y/z) and https://a.b <!-- note --> at c94928e."
    )
    assert common.anonymise(raw) == "Fixes #N by @someone, see bench and <link>  at <commit>."


def test_discrimination_scores_picks_of_the_candidate_and_keeps_the_key_out_of_prompts() -> None:
    def pick_by_content(messages: list[dict[str, str]]) -> object:
        text = messages[-1]["content"]
        # Find the letter of the description containing the draft's marker.
        blocks = text.split("## Description ")[1:]
        letter = next(b[0] for b in blocks if "DRAFT" in b)
        return tool_call("submit_pick", {"letter": letter, "tells": ["says DRAFT"]})

    model = FakeChatModel(script=[pick_by_content, pick_by_content])  # type: ignore[list-item]
    result = discriminate(
        model,
        "judge",
        ("ENH: x", "DRAFT body"),
        [("DOC: a", "real one"), ("BUG: b", "real two")],
        shuffles=2,
        seed=3,
        caps=CAPS,
        cache_key="k",
    )
    assert result.picked == 2 and result.valid == 2 and result.rate == 1.0
    assert result.chance == pytest.approx(1 / 3) and result.tells() == ["says DRAFT", "says DRAFT"]
    for req in model.requests:
        assert "candidate" not in req[-1]["content"] and "decoy" not in req[-1]["content"]


def test_novelty_needs_a_search_log_claim_citing_a_real_attempt() -> None:
    def claims(items: list[dict[str, object]]) -> FakeChatModel:
        return FakeChatModel(script=[tool_call("submit_claims", {"claims": items})])

    good = judge_novelty(
        claims(
            [
                {"text": "uses a bitset", "source": "from_diff"},
                {
                    "text": "a dense matrix was slower on sparse graphs",
                    "source": "from_search_log",
                    "attempt": 2,
                },
            ]
        ),
        "p",
        "t",
        "b",
        "dossier",
        attempts=(2,),
        require_search_log=True,
        caps=CAPS,
        cache_key="k",
    )
    assert good.ok

    bad = judge_novelty(
        claims(
            [
                {"text": "uses a bitset", "source": "from_diff"},
                {"text": "tried numba", "source": "from_search_log", "attempt": 9},
                {"text": "helps every user", "source": "unsupported"},
            ]
        ),
        "p",
        "t",
        "b",
        "dossier",
        attempts=(2,),
        require_search_log=True,
        caps=CAPS,
        cache_key="k",
    )
    problems = " ".join(bad.problems)
    assert not bad.ok and "unsupported: helps every user" in problems
    assert (
        "not in the search: tried numba" in problems
        and "says nothing the search taught" in problems
    )
