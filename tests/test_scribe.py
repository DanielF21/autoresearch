"""The Scribe against fakes: the filter, merged pull requests, the conversation, the CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from autoresearch.model.fake_model import FakeChatModel, text, tool_call
from autoresearch.scribe import candidates, cli, prs, scribe
from tests.scribe_helpers import git_source, make_run

PATCH = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n-LIMIT = 64\n+LIMIT = 4096\n x = 1\n"
)
OTHER = PATCH.replace("4096", "1024")


def raw_pr(
    number: int, merged: str | None, login: str = "dev", is_bot: bool = False
) -> dict[str, object]:
    return {
        "number": number,
        "title": f"ENH: change {number}",
        "body": None if number == 3 else f"Merged body {number}.",
        "author": {"login": login, "is_bot": is_bot},
        "mergedAt": merged,
        "url": f"https://github.com/o/r/pull/{number}",
    }


def pool() -> list[dict[str, object]]:
    return [
        raw_pr(1, "2026-09-01T00:00:00Z"),
        raw_pr(2, "2026-09-09T00:00:00Z", login="app/dependabot"),
        raw_pr(3, "2026-09-05T00:00:00Z"),
        raw_pr(4, "2026-09-03T00:00:00Z", login="precommit", is_bot=True),
        raw_pr(5, "2026-09-08T00:00:00Z"),
        raw_pr(6, None),
        raw_pr(7, "2026-09-07T00:00:00Z", login="renovate[bot]"),
    ]


@dataclass
class FakeGh:
    items: list[dict[str, object]]
    calls: list[list[str]] = field(default_factory=list)

    def run(self, args: list[str]) -> str:
        self.calls.append(args)
        return json.dumps(self.items)


def clean_run(tmp_path: Path, sha: str = "a" * 40) -> Path:
    return make_run(
        tmp_path,
        [
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}},
            {"patch": OTHER, "speedups": {"dense": 2.0, "sparse": 1.2}},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 0.5}},
        ],
        sha=sha,
    )


# ----- candidates and the filter --------------------------------------------------------


def test_candidates_flatten_measurements_and_read_both_setup_shapes(tmp_path: Path) -> None:
    run = make_run(
        tmp_path, [{"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}}, {"patch": None}]
    )
    target = candidates.read_target(run)
    assert target.setups == {"dense": "G = make(100, 0.5)", "sparse": "G = make(100, 0.01)"}
    first, second = candidates.load_candidates(run)
    assert first.number == 1 and first.clears_noise and first.tests_pass
    assert first.speedup == pytest.approx((3.0 * 1.5) ** 0.5)
    assert first.inputs[0].clean_pairs == 6 and first.inputs[0].base_median_s == 1.0
    assert second.patch is None and not second.measured


def test_the_filter_keeps_clean_candidates_and_records_every_reason(tmp_path: Path) -> None:
    run = make_run(
        tmp_path,
        [
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 0.5}},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}, "duplicate_of": "0001"},
            {"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 1.5}, "tests_ok": False},
            {"patch": None},
        ],
    )
    kept, excluded = candidates.filter_candidates(candidates.load_candidates(run))
    assert [c.number for c in kept] == [1]
    assert any("regresses on: sparse" in r for r in excluded[2])
    assert any("duplicate of attempt 0001" in r for r in excluded[3])
    assert any("tests did not pass" in r for r in excluded[4])
    assert excluded[5] == ("no patch (no_patch)",)


def test_describe_shows_the_diff_measurements_tests_and_rationale(tmp_path: Path) -> None:
    (c, *_) = candidates.load_candidates(clean_run(tmp_path))
    text_ = candidates.describe(c)
    assert "## Attempt 1" in text_ and "2 lines changed." in text_ and "+LIMIT = 4096" in text_
    assert "| dense | `G = make(100, 0.5)` | 3.00x |" in text_
    assert "module suite: 56 passed" in text_ and "made it faster" in text_


# ----- merged pull requests -------------------------------------------------------------


def test_merged_prs_drop_bots_sort_by_merge_time_and_truncate() -> None:
    got = prs.newest_by_people(pool(), 2)
    assert [p.number for p in got] == [5, 3]
    assert got[1].body == "" and got[0].author == "dev"


def test_fetch_asks_gh_for_a_larger_pool_and_refuses_too_few() -> None:
    gh = FakeGh(pool())
    assert [p.number for p in prs.fetch_merged_prs(gh, "o/r", 3)] == [5, 3, 1]
    assert gh.calls[0][:8] == ["pr", "list", "--repo", "o/r", "--state", "merged", "--limit", "9"]
    with pytest.raises(prs.GhError, match="only 3"):
        prs.fetch_merged_prs(FakeGh(pool()), "o/r", 4)


def test_github_repo_and_saved_prs_round_trip(tmp_path: Path) -> None:
    assert prs.github_repo("https://github.com/networkx/networkx") == "networkx/networkx"
    assert prs.github_repo("git@github.com:o/r.git") == "o/r"
    with pytest.raises(ValueError, match="--repo"):
        prs.github_repo("https://example.invalid/o/r")
    saved = prs.newest_by_people(pool(), 3)
    prs.save_prs(tmp_path / "prs.json", saved)
    assert prs.load_prs(tmp_path / "prs.json") == saved


# ----- the conversation -----------------------------------------------------------------


def test_the_model_picks_then_writes_in_the_same_conversation(tmp_path: Path) -> None:
    run = clean_run(tmp_path)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    model = FakeChatModel(
        script=[
            tool_call("submit_pick", {"attempt": 3, "reasons": "fast"}),
            tool_call("submit_pick", {"attempt": 1, "reasons": "fastest on both inputs"}),
            tool_call("submit_pr", {"title": "ENH: faster table", "body": "Body text."}),
        ]
    )
    out = tmp_path / "out"
    out.mkdir()
    draft = scribe.run(
        model, candidates.read_target(run), kept, prs.newest_by_people(pool(), 3), out
    )

    assert draft == scribe.Draft(1, "fastest on both inputs", "ENH: faster table", "Body text.")
    first = model.requests[0]
    assert first[0]["content"] == scribe.PROMPT.read_text()
    assert "## Attempt 1" in first[1]["content"] and "## Attempt 2" in first[1]["content"]
    assert "## Attempt 3" not in first[1]["content"]
    assert "attempt must be one of [1, 2] or null" in model.requests[1][-1]["content"]
    write = model.requests[2]
    assert write[1] == first[1]  # same conversation: the candidates are still there
    assert "attempt 1" in write[-1]["content"] and "Merged body 5." in write[-1]["content"]
    assert len(set(model.cache_keys)) == 1

    assert json.loads((out / "pick.json").read_text())["attempt"] == 1
    assert (out / "body.md").read_text() == "Body text.\n"
    assert (out / "title.txt").read_text() == "ENH: faster table\n"
    assert (out / "pr.md").read_text() == "# ENH: faster table\n\nBody text.\n"
    assert (out / "patch.diff").read_text() == PATCH
    assert json.loads((out / "usage.json").read_text())["prompt_tokens"] == 300
    assert len(json.loads((out / "messages.json").read_text())) == len(write) + 2


def test_declining_every_candidate_writes_no_pull_request(tmp_path: Path) -> None:
    run = clean_run(tmp_path)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    model = FakeChatModel(
        script=[tool_call("submit_pick", {"attempt": None, "reasons": "private attribute"})]
    )
    draft = scribe.run(model, candidates.read_target(run), kept, [], tmp_path, prompt="p")
    assert draft.pick is None and draft.reasons == "private attribute"
    assert len(model.requests) == 1 and not (tmp_path / "body.md").exists()
    assert not (tmp_path / "pr.md").exists() and not (tmp_path / "patch.diff").exists()
    assert json.loads((tmp_path / "pick.json").read_text()) == {
        "attempt": None,
        "reasons": "private attribute",
    }


def test_replies_without_a_valid_call_stop_after_the_retries(tmp_path: Path) -> None:
    run = clean_run(tmp_path)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    model = FakeChatModel(script=[text("hmm")] * scribe.TRIES)
    with pytest.raises(scribe.ScribeError, match="no valid submit_pick"):
        scribe.run(model, candidates.read_target(run), kept, [], tmp_path, prompt="p")
    assert "Answer by calling submit_pick." in model.requests[1][-1]["content"]
    assert (tmp_path / "messages.json").exists() and not (tmp_path / "pick.json").exists()


def test_reads_of_the_source_are_answered_and_are_not_misses(tmp_path: Path) -> None:
    src, sha = git_source(tmp_path)
    run = clean_run(tmp_path, sha)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    reads = [
        tool_call("read_file", {"path": "pkg/mod.py"}, f"r{n}") for n in range(scribe.TRIES + 1)
    ]
    model = FakeChatModel(
        script=[
            *reads,
            tool_call("grep", {"pattern": "LIMIT"}, "g1"),
            tool_call("read_file", {"path": "../../etc/passwd"}, "bad"),
            tool_call("submit_pick", {"attempt": 1, "reasons": "rebuilt per call"}),
            tool_call("submit_pr", {"title": "t", "body": "b"}),
        ]
    )
    draft = scribe.run(
        model, candidates.read_target(run), kept, [], tmp_path, prompt="p", source=src
    )
    assert draft.pick == 1
    replies = {m["tool_call_id"]: m["content"] for m in model.requests[-1] if m["role"] == "tool"}
    assert "the table is rebuilt on every call" in replies["r0"]
    assert "pkg/mod.py:2: LIMIT = 64" in replies["g1"]
    assert replies["bad"].startswith("error:") and "outside" in replies["bad"]


def test_reads_past_the_limit_are_refused_and_count_as_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, sha = git_source(tmp_path)
    run = clean_run(tmp_path, sha)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    monkeypatch.setattr(scribe, "MAX_READS", 2)
    model = FakeChatModel(
        script=[
            tool_call("read_file", {"path": "pkg/mod.py"}, f"r{n}") for n in range(2 + scribe.TRIES)
        ]
    )
    with pytest.raises(scribe.ScribeError, match="no valid submit_pick"):
        scribe.run(model, candidates.read_target(run), kept, [], tmp_path, prompt="p", source=src)
    assert "all 2 reads are used" in model.requests[-1][-1]["content"]


def test_without_a_source_a_read_is_refused(tmp_path: Path) -> None:
    run = clean_run(tmp_path)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    model = FakeChatModel(
        script=[
            tool_call("read_file", {"path": "pkg/mod.py"}, "r0"),
            tool_call("submit_pick", {"attempt": None, "reasons": "none"}),
        ]
    )
    scribe.run(model, candidates.read_target(run), kept, [], tmp_path, prompt="p")
    assert "the tools here are submit_pick" in model.requests[1][-1]["content"]


def test_hyphens_and_dashes_are_found_in_prose_but_not_in_code_or_markdown() -> None:
    body = (
        "A full-graph path \u2014 cached, 1.3 s -> 0.3 s.\n\n"
        "- a list item\n\n"
        "| input | base |\n| --- | :---: |\n| `x - 1` | 2 |\n\n"
        "```python\ny = a - b\n```\n"
        "In-or-out sets, full-graph again."
    )
    assert scribe.prose_dashes(body) == ["full-graph", "\u2014", "->", "In-or-out"]
    assert scribe.prose_dashes("| a |\n|---|---|\n- item `a-b`") == []
    problem = scribe._validate_pr("")({"title": "Set-based speedup", "body": "Plain."})
    assert problem is not None and "Set-based" in problem and "space instead" in problem
    assert scribe._validate_pr("")({"title": "Set based speedup", "body": "Plain."}) is None


def test_numbers_with_units_must_come_from_the_evidence() -> None:
    evidence = "| dense | 6.23x | 1.0218 | 12.3 ms |\nGeometric mean: 3.00x."
    body = (
        "The call is 6.23x faster on dense, 6.2x rounded, 6x coarse, taking 12 ms. "
        "It walks 1000 nodes. Claimed 7.1x elsewhere and 35% less time, 7.1x again."
    )
    assert scribe.unsupported_numbers(body, evidence) == ["7.1x", "35%"]
    assert scribe.unsupported_numbers("0x1f and 3.00 x", evidence) == []


def test_a_body_with_an_unmeasured_number_goes_back_to_the_model(tmp_path: Path) -> None:
    run = clean_run(tmp_path)
    kept, _ = candidates.filter_candidates(candidates.load_candidates(run))
    model = FakeChatModel(
        script=[
            tool_call("submit_pick", {"attempt": 1, "reasons": "fastest"}),
            tool_call("submit_pr", {"title": "t", "body": "About 9.9x faster."}, "p1"),
            tool_call("submit_pr", {"title": "t", "body": "3.00x faster on dense."}, "p2"),
        ]
    )
    draft = scribe.run(model, candidates.read_target(run), kept, [], tmp_path, prompt="p")
    assert draft.body == "3.00x faster on dense."
    assert "not in the measurements you were shown: 9.9x" in model.requests[2][-1]["content"]


def test_the_shortlist_keeps_the_fastest_half_and_the_smallest_of_the_rest(
    tmp_path: Path,
) -> None:
    (base, *_) = candidates.load_candidates(clean_run(tmp_path))
    header = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
    many = [
        replace(base, number=n, speedup=float(n), patch=header + "+x\n" * n) for n in range(1, 31)
    ]
    kept, left = candidates.shortlist(many)
    half = candidates.MAX_CANDIDATES // 2
    assert [c.number for c in kept] == [*range(1, half + 1), *range(31 - half, 31)]
    assert sorted(left) == list(range(half + 1, 31 - half))
    assert all(r[0].startswith("not shortlisted") for r in left.values())
    assert candidates.shortlist(many[:3]) == (many[:3], {})


def test_worker_config_comes_from_the_run_with_an_optional_model(tmp_path: Path) -> None:
    run = clean_run(tmp_path)
    cfg = scribe.worker_config(run)
    assert cfg.model == "worker/model" and cfg.turn_timeout == 600
    assert scribe.worker_config(run, "other/model").model == "other/model"


# ----- cli ------------------------------------------------------------------------------


def test_cli_with_nothing_kept_calls_neither_gh_nor_a_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_run(tmp_path, [{"patch": PATCH, "speedups": {"dense": 3.0, "sparse": 0.5}}])
    model, gh = FakeChatModel(), FakeGh(pool())
    rc = cli.main([str(run), "--out", str(tmp_path / "out")], model=model, gh=gh)
    assert rc == 1 and model.requests == [] and gh.calls == []
    assert "No model was called" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_cli_end_to_end_with_fakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src, sha = git_source(tmp_path)
    run = clean_run(tmp_path, sha)
    monkeypatch.setattr(candidates, "MAX_CANDIDATES", 2)
    model = FakeChatModel(
        script=[
            tool_call("submit_pick", {"attempt": 2, "reasons": "smaller"}),
            tool_call("submit_pr", {"title": "ENH: t", "body": "b"}),
        ]
    )
    gh = FakeGh(pool())
    argv = [str(run), "--n", "2", "--out", str(tmp_path / "out"), "--source", str(src)]
    rc = cli.main(argv, model=model, gh=gh)
    assert rc == 0 and gh.calls[0][3] == "o/r"
    (out,) = (tmp_path / "out" / "fake_run").iterdir()
    assert (out / "body.md").read_text() == "b\n"
    assert (out / "pr.md").read_text() == "# ENH: t\n\nb\n"
    assert (out / "patch.diff").read_text() == OTHER
    assert [p["number"] for p in json.loads((out / "prs.json").read_text())] == [5, 3]
    assert "regresses on: sparse" in json.loads((out / "excluded.json").read_text())["0003"][0]
    assert "Picked attempt 2: smaller" in capsys.readouterr().out


def test_cli_stops_before_the_model_when_the_source_is_at_another_commit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src, _ = git_source(tmp_path)
    run = clean_run(tmp_path)
    model = FakeChatModel()
    argv = [str(run), "--n", "2", "--out", str(tmp_path / "out"), "--source", str(src)]
    assert cli.main(argv, model=model, gh=FakeGh(pool())) == 1
    assert model.requests == [] and "the run is at aaaaaaaaaaaa" in capsys.readouterr().out
