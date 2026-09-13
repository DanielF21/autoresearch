"""Fetching, freezing and describing the style corpus, against a fake gh."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autoresearch.scribe import corpus, corpus_stats
from tests.scribe_helpers import pr

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def raw_pr(
    number: int, merged: str | None, login: str = "dev", is_bot: bool = False
) -> dict[str, object]:
    return {
        "number": number,
        "title": f"ENH: change {number}",
        "body": None if number == 3 else f"Body of {number}.",
        "author": {"login": login, "is_bot": is_bot},
        "labels": [{"name": "type: Enhancements"}],
        "additions": 5,
        "deletions": 1,
        "files": [{"path": "pkg/mod.py", "additions": 5, "deletions": 1}],
        "mergedAt": merged,
        "url": f"https://example.invalid/pull/{number}",
    }


@dataclass
class FakeGh:
    prs: list[dict[str, object]]
    template: str = "<!-- template -->\n"
    calls: list[list[str]] = field(default_factory=list)

    def run(self, args: list[str]) -> str:
        self.calls.append(args)
        if args[:2] == ["pr", "list"]:
            return json.dumps(self.prs)
        if args[0] == "api":
            if not self.template:
                raise corpus.GhError("404")
            return self.template
        raise AssertionError(args)


def pool() -> list[dict[str, object]]:
    return [
        raw_pr(1, "2026-09-01T00:00:00Z"),
        raw_pr(2, "2026-09-09T00:00:00Z", login="app/dependabot"),
        raw_pr(3, "2026-09-05T00:00:00Z"),
        raw_pr(4, "2026-09-03T00:00:00Z", login="precommit", is_bot=True),
        raw_pr(5, "2026-09-08T00:00:00Z"),
        raw_pr(6, None),
        raw_pr(7, "2026-09-07T00:00:00Z", login="renovate[bot]"),
        raw_pr(8, "2026-09-02T00:00:00Z", login="blocked"),
        raw_pr(9, "2026-09-06T00:00:00Z"),
    ]


def test_select_drops_bots_sorts_by_merge_time_and_truncates() -> None:
    prs, excluded = corpus.select_prs(pool(), n=3, exclude=("blocked",))
    assert [p.number for p in prs] == [5, 9, 3]
    assert {e["number"] for e in excluded} == {2, 4, 7, 8}
    assert prs[2].body == ""
    assert prs[0].labels == ("type: Enhancements",) and prs[0].files == ("pkg/mod.py",)


def test_fetch_freezes_a_verified_corpus_and_refuses_to_overwrite(tmp_path: Path) -> None:
    gh = FakeGh(pool())
    kwargs = dict(
        repo="o/r",
        n=4,
        pool=50,
        exclude_authors=(),
        holdout_n=2,
        seed=7,
        template_path=".github/PULL_REQUEST_TEMPLATE.md",
        out_root=tmp_path,
        now=NOW,
        stats=corpus_stats.corpus_stats,
    )
    out = corpus.fetch_corpus(gh, **kwargs)  # type: ignore[arg-type]
    assert gh.calls[0] == [
        "pr",
        "list",
        "--repo",
        "o/r",
        "--state",
        "merged",
        "--limit",
        "50",
        "--json",
        corpus.PR_FIELDS,
    ]
    loaded = corpus.load_corpus(out)
    # 8's author is excluded only by config, and this call excludes nobody.
    assert [p.number for p in loaded.prs] == [5, 9, 3, 8]
    assert len(loaded.holdout) == 2 and len(loaded.exemplars) == 2
    assert not {p.number for p in loaded.holdout} & {p.number for p in loaded.exemplars}
    assert loaded.template == "<!-- template -->\n"
    assert loaded.stats["n"] == 4

    with pytest.raises(FileExistsError):
        corpus.fetch_corpus(FakeGh(pool()), **kwargs)  # type: ignore[arg-type]

    (out / "prs" / "5.json").write_text("{}")
    with pytest.raises(corpus.CorpusError, match="changed or missing"):
        corpus.load_corpus(out)


def test_fetch_refuses_a_pool_too_small_for_n(tmp_path: Path) -> None:
    with pytest.raises(corpus.GhError, match="raise the pool"):
        corpus.fetch_corpus(
            FakeGh(pool()),
            repo="o/r",
            n=10,
            pool=9,
            exclude_authors=(),
            holdout_n=2,
            seed=1,
            template_path="t.md",
            out_root=tmp_path,
            now=NOW,
        )


def test_split_is_deterministic_and_disjoint() -> None:
    nums = list(range(20))
    a = corpus.split(nums, 8, seed=3)
    assert a == corpus.split(nums, 8, seed=3)
    assert len(a[1]) == 8 and set(a[0]).isdisjoint(a[1])
    assert a != corpus.split(nums, 8, seed=4)


def test_title_prefixes_follow_the_conventions_seen_in_real_titles() -> None:
    assert corpus_stats.title_prefix("BUG: fix x") == "BUG"
    assert corpus_stats.title_prefix("[BUG][PERF] Fix ISMAGS") == "BUG+PERF"
    assert corpus_stats.title_prefix("fix: handle empty sets") == "FIX"
    assert corpus_stats.title_prefix("Expose Matcher classes") == ""


def test_body_stats_ignore_markup_inside_code_fences() -> None:
    body = "Intro with `code` and #123.\n\n- one\n- two\n\n```python\n# not a header\n- not a bullet\n```\n"
    st = corpus_stats.body_stats(body)
    assert st.headers == 0 and st.bullet_lines == 2 and st.code_fences == 1
    assert st.inline_code == 1 and st.refs == 1 and st.paragraphs == 3


def test_corpus_stats_group_by_prefix() -> None:
    stats = corpus_stats.corpus_stats(
        [
            pr(1, "DOC: a", "short"),
            pr(2, "BUG: b", "a much longer body " * 10),
            pr(3, "DOC: c", "mid length"),
        ]
    )
    assert stats["prefixes"] == {"DOC": 2, "BUG": 1}
    assert stats["by_prefix"]["DOC"]["n"] == 2
    assert stats["body"]["chars"]["min"] == 5
