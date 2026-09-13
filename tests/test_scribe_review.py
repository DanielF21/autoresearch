"""The agent loop, the method bundle, the reviewer, and the frontier."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autoresearch.model.fake_model import FakeChatModel, text, tool_call
from autoresearch.model.protocol import ModelError
from autoresearch.scribe import facts as facts_mod
from autoresearch.scribe import method as method_mod
from autoresearch.scribe import review, runread
from autoresearch.scribe.loop import Caps, LoopStop, run_agent
from autoresearch.scribe.repo import RepoCache
from autoresearch.scribe.select import Point, pareto_frontier
from autoresearch.scribe.tools import READ_TOOLS, Roots, ToolContext, submit_tool
from tests.scribe_helpers import diff_in, make_run, make_upstream

CAPS = Caps(max_turns=6, max_seconds=600, max_input_tokens=10_000_000)
V1 = method_mod.PACKAGED / "v1"
SOURCE = "def f(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total\n"
FAST = "def f(xs):\n    return sum(xs)\n"


def answer_tool() -> tuple:  # type: ignore[type-arg]
    return (
        *READ_TOOLS,
        submit_tool(
            "submit",
            "done",
            {"type": "object", "properties": {"n": {"type": "integer"}}},
            lambda a: None if a.get("n") == 3 else "n must be 3",
        ),
    )


def ctx_for(tmp_path: Path) -> ToolContext:
    (tmp_path / "f.txt").write_text("hello\n")
    return ToolContext(roots=Roots(dirs={"base": tmp_path}))


def test_loop_reads_then_retries_a_rejected_submit_then_ends(tmp_path: Path) -> None:
    model = FakeChatModel(
        script=[
            tool_call("read_file", {"path": "base:f.txt"}),
            tool_call("submit", {"n": 2}),
            tool_call("submit", {"n": 3}),
        ]
    )
    messages = [{"role": "user", "content": "go"}]
    r = run_agent(model, messages, answer_tool(), ctx_for(tmp_path), CAPS, cache_key="k")
    assert r.stop is LoopStop.SUBMITTED and r.submitted == {"n": 3} and r.turns == 3
    assert "hello" in model.requests[1][-1]["content"]
    assert "not accepted: n must be 3" in model.requests[2][-1]["content"]
    assert r.usage.prompt_tokens == 300 and '"kind": "end"' in r.transcript


@pytest.mark.parametrize(
    ("script", "stop"),
    [
        (
            [tool_call("read_file", {"path": "base:f.txt", "start_line": i}) for i in range(1, 9)],
            LoopStop.MAX_TURNS,
        ),
        ([tool_call("read_file", {"path": "base:f.txt"})] * 3, LoopStop.REPEATED_TOOL_CALL),
        ([text("thinking"), text("still thinking")], LoopStop.NO_PROGRESS),
        ([ModelError("down")], LoopStop.MODEL_ERROR),
    ],
)
def test_loop_stops(tmp_path: Path, script: list, stop: LoopStop) -> None:  # type: ignore[type-arg]
    model = FakeChatModel(script=list(script))
    r = run_agent(
        model,
        [{"role": "user", "content": "go"}],
        answer_tool(),
        ctx_for(tmp_path),
        CAPS,
        cache_key="k",
    )
    assert r.stop is stop and r.submitted is None


def test_loop_nudges_once_before_giving_up(tmp_path: Path) -> None:
    model = FakeChatModel(script=[text("hmm"), tool_call("submit", {"n": 3})])
    r = run_agent(
        model,
        [{"role": "user", "content": "go"}],
        answer_tool(),
        ctx_for(tmp_path),
        CAPS,
        cache_key="k",
    )
    assert r.stop is LoopStop.SUBMITTED
    assert "not read" in model.requests[1][-1]["content"]


def test_the_packaged_method_loads_and_its_hash_tracks_its_bytes(tmp_path: Path) -> None:
    m = method_mod.load_method(V1)
    assert m.review_mode == "diff_only" and m.accept_rate is None
    assert set(m.prompts) == set(method_mod.PROMPTS) and m.slop
    copy = tmp_path / "v1"
    shutil.copytree(V1, copy)
    assert method_mod.load_method(copy).hash == m.hash
    (copy / "writer.md").write_text("changed\n")
    assert method_mod.load_method(copy).hash != m.hash


def test_method_refuses_a_missing_prompt(tmp_path: Path) -> None:
    copy = tmp_path / "v1"
    shutil.copytree(V1, copy)
    (copy / "judge_novelty.md").unlink()
    with pytest.raises(method_mod.MethodError, match="judge_novelty"):
        method_mod.load_method(copy)


def reviewed_setup(tmp_path: Path) -> tuple[runread.Candidate, facts_mod.Facts, ToolContext]:
    repo, sha = make_upstream(tmp_path, {"m.py": SOURCE})
    patch = diff_in(repo, "m.py", FAST)
    cache = RepoCache(tmp_path / "cache")
    base = cache.tree(f"file://{repo}", sha)
    patched = cache.tree(f"file://{repo}", sha, patch)
    run = make_run(
        tmp_path,
        [
            {
                "patch": patch,
                "speedups": {"dense": 2.0, "sparse": 1.5},
                "rationale": "SECRET RATIONALE",
            }
        ],
    )
    (cand,) = runread.load_candidates(run)
    f = facts_mod.compute_facts(patch, {"m.py": SOURCE}, {"m.py": FAST})
    ctx = ToolContext(roots=Roots(dirs={"base": base, "patched": patched}))
    return cand, f, ctx


def verdict_args(end: int) -> dict[str, object]:
    return {
        "verdict": "needs_changes",
        "cost_model": {"before": "O(n) Python loop", "after": "O(n) in C", "changed": False},
        "inputs_at_risk": [],
        "concerns": [
            {
                "kind": "readability",
                "claim": "the loop is gone",
                "evidence": [{"path": "patched:m.py", "start": 1, "end": end}],
            }
        ],
        "reasons": "fine but check types",
    }


def test_review_in_diff_only_mode_hides_measurements_and_checks_citations(tmp_path: Path) -> None:
    cand, f, ctx = reviewed_setup(tmp_path)
    m = method_mod.load_method(V1)
    model = FakeChatModel(
        script=[
            tool_call("read_file", {"path": "patched:m.py"}),
            tool_call("submit_verdict", {"verdict": "maybe"}),
            tool_call("submit_verdict", verdict_args(2)),
        ]
    )
    out = review.review_candidate(model, m, cand, f, ctx, CAPS, cache_key="k")
    assert out.valid and out.verdict is not None and out.verdict.verdict == "needs_changes"
    sent = model.requests[0][1]["content"]
    assert "return sum(xs)" in sent and "SECRET RATIONALE" not in sent and "noise floor" not in sent
    assert "verdict must be one of" in model.requests[2][-1]["content"]
    assert review.Verdict.from_dict(out.verdict.to_dict()) == out.verdict

    full = review.review_message(cand, f, "full", ctx.roots)
    assert "SECRET RATIONALE" in full and "noise floor" in full

    bad = FakeChatModel(script=[tool_call("submit_verdict", verdict_args(40))])
    worse = review.review_candidate(bad, m, cand, f, ctx, CAPS, cache_key="k")
    assert not worse.valid and "outside 1 to 2" in worse.evidence_problems[0]


def test_frontier_on_the_t1_w4c_numbers_keeps_the_fastest_and_the_smallest() -> None:
    """Geomeans and added plus removed lines of t1_w4c attempts 1 to 4."""
    points = [Point(1, 6.10, 75), Point(2, 6.31, 98), Point(3, 5.02, 103), Point(4, 6.12, 65)]
    assert [p.number for p in pareto_frontier(points)] == [2, 4]
    assert pareto_frontier([Point(1, 2.0, 10), Point(2, 2.0, 10)]) == (
        Point(1, 2.0, 10),
        Point(2, 2.0, 10),
    )


def test_compare_rejects_an_attempt_that_was_not_offered(tmp_path: Path) -> None:
    cand, f, ctx = reviewed_setup(tmp_path)
    v = review.Verdict.from_dict({**verdict_args(2), "verdict": "mergeable"})
    model = FakeChatModel(
        script=[
            tool_call("submit_pick", {"attempt": 99, "reasons": "x"}),
            tool_call("submit_pick", {"attempt": 1, "reasons": "smaller"}),
        ]
    )
    out = review.compare_candidates(
        model, method_mod.load_method(V1), [(cand, f, v)], ctx, CAPS, cache_key="k"
    )
    assert out.pick == 1 and out.reasons == "smaller"
    assert "attempt must be one of [1]" in model.requests[1][-1]["content"]
