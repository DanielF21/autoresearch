"""The production draft loop end to end, the dev tools, and the CLI, all against fakes."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest

from autoresearch.model.fake_model import FakeChatModel, Scripted, tool_call
from autoresearch.model.protocol import Message, ModelResponse
from autoresearch.scribe import cli, runread
from autoresearch.scribe import corpus as corpus_mod
from autoresearch.scribe import facts as facts_mod
from autoresearch.scribe import method as method_mod
from autoresearch.scribe.corpus_stats import corpus_stats
from autoresearch.scribe.dev import blind, calibrate
from autoresearch.scribe.dev.harness import load_devset
from autoresearch.scribe.dev.review_check import load_labels
from autoresearch.scribe.layout import append_jsonl
from autoresearch.scribe.loop import Caps
from autoresearch.scribe.tools import Roots, ToolContext
from autoresearch.scribe.write import RoleCaps, Roles, WriteInputs, write_for_candidate
from tests.scribe_helpers import make_run

ROOT = Path(__file__).parent.parent
SCRIBE = ROOT / "src" / "autoresearch" / "scribe"
V1_DIR = method_mod.PACKAGED / "v1"
CAPS = Caps(5, 600, 10_000_000)
PATCH = (
    "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n"
    "@@ -1,2 +1,2 @@\n-LIMIT = 64\n+LIMIT = 4096\n x = 1\n"
)
TITLE = "ENH: Grow the lookup table in pkg.mod"
GOOD = (
    "The lookup table in `pkg/mod.py` grows from 64 to 4096 entries, so the common case "
    "stops falling back to the slow path.\n\n"
    "Against the base commit, dense graphs (`make(100, 0.5)`) run 6.31x faster and sparse "
    "graphs (`make(100, 0.01)`) run 2.61x faster. Both inputs computed the same result as "
    "before, and the full suite passes with 9090 tests.\n\n"
    "A smaller table was also measured and made sparse graphs slower, so the size is what "
    "matters here."
)
BAD = GOOD.replace("6.31x", "7.2x")


# ----- production never reaches dev -----------------------------------------------------


def _scribe_imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("autoresearch.scribe")
        ):
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("autoresearch.scribe"))
    return out


def _module_path(name: str) -> Path | None:
    rel = name.removeprefix("autoresearch.scribe").strip(".").replace(".", "/")
    base = SCRIBE / rel if rel else SCRIBE
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    return None


def test_the_production_path_never_imports_the_dev_package() -> None:
    seen: set[str] = set()
    stack = ["autoresearch.scribe.write"]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(name)
        if path is not None:
            stack.extend(_scribe_imports(path))
    assert "autoresearch.scribe.review" in seen and "autoresearch.scribe.checks" in seen
    assert sorted(n for n in seen if n.startswith("autoresearch.scribe.dev")) == []


# ----- fixtures -------------------------------------------------------------------------


class ListGh:
    def __init__(self, prs: list[dict[str, object]]) -> None:
        self.prs = prs

    def run(self, args: list[str]) -> str:
        return json.dumps(self.prs) if args[0] == "pr" else ""


def make_corpus(tmp_path: Path, n: int = 8, holdout: int = 4) -> corpus_mod.Corpus:
    sentence = "The previous implementation walked every entry again on each call. "
    raw = [
        {
            "number": 100 + i,
            "title": f"ENH: speed up the lookup path in module number {i}",
            "body": sentence * (3 + 2 * i),
            "author": {"login": f"dev{i}", "is_bot": False},
            "labels": [],
            "additions": 5,
            "deletions": 1,
            "files": [{"path": "pkg/mod.py"}],
            "mergedAt": f"2026-09-{i:02d}T00:00:00Z",
            "url": "",
        }
        for i in range(1, n + 1)
    ]
    out = corpus_mod.fetch_corpus(
        ListGh(raw),
        repo="o/r",
        n=n,
        pool=n,
        exclude_authors=(),
        holdout_n=holdout,
        seed=1,
        template_path="t.md",
        out_root=tmp_path / "corpus",
        stats=corpus_stats,
    )
    return corpus_mod.load_corpus(out)


def small_method(tmp_path: Path) -> method_mod.WriterMethod:
    d = tmp_path / "method"
    shutil.copytree(V1_DIR, d)
    toml = d / "method.toml"
    toml.write_text(toml.read_text().replace("decoys = 3", "decoys = 2"))
    return method_mod.load_method(d)


def judge_reply(messages: list[Message]) -> ModelResponse:
    if "against the record it was written from" in messages[0]["content"]:
        return tool_call(
            "submit_claims",
            {
                "claims": [
                    {"text": "the table grows", "source": "from_diff"},
                    {
                        "text": "a smaller table was slower",
                        "source": "from_search_log",
                        "attempt": 2,
                    },
                ]
            },
        )
    return tool_call("submit_pick", {"letter": "A", "tells": ["uniform rhythm"]})


def write_inputs(tmp_path: Path) -> WriteInputs:
    run = make_run(
        tmp_path,
        [
            {"patch": PATCH, "speedups": {"dense": 6.31, "sparse": 2.61}},
            {"patch": PATCH.replace("4096", "9"), "speedups": {"dense": 3.0, "sparse": 0.5}},
        ],
    )
    first, second = runread.load_candidates(run)
    return WriteInputs(
        candidate=first,
        facts=facts_mod.compute_facts(PATCH),
        target=runread.read_target(run),
        others=[(second, facts_mod.compute_facts(second.patch or ""))],
        ctx=ToolContext(Roots(dirs={"base": tmp_path})),
    )


# ----- the draft loop -------------------------------------------------------------------


def test_draft_loop_discards_a_bad_draft_critiques_the_best_and_resumes_without_calls(
    tmp_path: Path,
) -> None:
    corpus = make_corpus(tmp_path)
    method = small_method(tmp_path)
    inputs = write_inputs(tmp_path)
    script: list[Scripted] = [tool_call("submit_body", {"title": TITLE, "body": BAD})]
    script += [tool_call("submit_body", {"title": TITLE, "body": GOOD}) for _ in range(5)]
    writer = FakeChatModel(script=script)
    judge = FakeChatModel(script=[judge_reply] * 40)
    roles = Roles(reviewer=FakeChatModel(), writer=writer, judge=judge)
    caps = RoleCaps(CAPS, CAPS, CAPS)
    out = tmp_path / "write"

    outcome = write_for_candidate(
        out, inputs, corpus, method, roles, caps, seed=7, allow_uncalibrated=True
    )

    assert outcome.status == "uncalibrated"
    assert (out / "final_body.md").read_text().strip() == GOOD
    assert (out / "final_title.txt").read_text().strip() == TITLE
    bad = json.loads((out / "iter_01" / "draft_01" / "draft.json").read_text())
    tracing = next(c for c in bad["checks"] if c["name"] == "numbers trace to the record")
    assert not tracing["ok"] and tracing["detail"][0].startswith("7.2x")
    assert bad["discrimination"] is None  # never judged
    good = json.loads((out / "iter_01" / "draft_02" / "draft.json").read_text())
    assert good["gates_ok"], good["checks"]
    assert "Critique of the best draft so far" in writer.requests[2][1]["content"]
    dossier = (out / "dossier.md").read_text()
    assert "attempt 2 [same file]" in dossier and "crossover" not in dossier
    for req in judge.requests:
        assert "decoy" not in req[-1]["content"] and "candidate" not in req[-1]["content"]

    idle = Roles(reviewer=FakeChatModel(), writer=FakeChatModel(), judge=FakeChatModel())
    again = write_for_candidate(
        out, inputs, corpus, method, idle, caps, seed=7, allow_uncalibrated=True
    )
    assert again.status == "uncalibrated"
    assert idle.writer.requests == [] and idle.judge.requests == []  # type: ignore[attr-defined]


def test_without_a_calibrated_rate_production_rejects_unless_told(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    writer = FakeChatModel(
        script=[tool_call("submit_body", {"title": TITLE, "body": GOOD}) for _ in range(6)]
    )
    roles = Roles(FakeChatModel(), writer, FakeChatModel(script=[judge_reply] * 40))
    outcome = write_for_candidate(
        tmp_path / "w",
        write_inputs(tmp_path),
        corpus,
        small_method(tmp_path),
        roles,
        RoleCaps(CAPS, CAPS, CAPS),
        seed=7,
        allow_uncalibrated=False,
    )
    assert outcome.status == "rejected" and not (tmp_path / "w" / "final_body.md").exists()


def test_a_writer_that_never_passes_the_gates_abstains(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    writer = FakeChatModel(
        script=[tool_call("submit_body", {"title": TITLE, "body": BAD}) for _ in range(6)]
    )
    roles = Roles(FakeChatModel(), writer, FakeChatModel())
    outcome = write_for_candidate(
        tmp_path / "w",
        write_inputs(tmp_path),
        corpus,
        small_method(tmp_path),
        roles,
        RoleCaps(CAPS, CAPS, CAPS),
        seed=7,
        allow_uncalibrated=True,
    )
    assert outcome.status == "abstain"
    assert json.loads((tmp_path / "w" / "outcome.json").read_text())["reasons"] == [
        "no draft passed every gate"
    ]


# ----- dev tools ------------------------------------------------------------------------


def test_blind_packet_hides_the_key_and_unblind_scores_a_ranking(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    dev = tmp_path / "dev"
    for i, rate in enumerate([0.0, 0.5, 1.0]):
        d = dev / f"v{i}"
        d.mkdir(parents=True)
        (d / "final_title.txt").write_text(f"ENH: draft {i}\n")
        (d / "final_body.md").write_text(f"Draft body {i}, see https://x.y/z\n")
        append_jsonl(
            dev / "scores.jsonl",
            {
                "status": "uncalibrated",
                "dir": str(d),
                "method_hash": "m" * 40,
                "run": "runs/t1",
                "attempt": i + 1,
                "final_rate": rate,
            },
        )
    packet = blind.build_packet(dev, corpus, real_count=2, seed=3)
    for p in packet.iterdir():
        text = p.read_text()
        assert "machine:" not in text and "real:" not in text and "https://x.y" not in text

    with pytest.raises(ValueError, match="rank every item"):
        blind.unblind(packet)

    key = json.loads(next((dev / blind.KEYS_DIR).glob("*.json")).read_text())
    items = key["items"]
    machine = sorted(
        (label for label in items if items[label].startswith("machine:")),
        key=lambda label: key["judge_final_rates"][items[label]],
    )
    real = [label for label in sorted(items) if label not in machine]
    order = real + machine
    (packet / "ranking.toml").write_text(
        "[ranking]\n" + "".join(f'"{label}" = {i + 1}\n' for i, label in enumerate(order))
    )
    result = blind.unblind(packet)
    assert result["mean_real_rank"] == 1.5 and result["best_machine_rank"] == 3
    assert result["judge_rate_vs_rank_spearman"] == pytest.approx(1.0)
    assert result["machine_ranked_above_some_real"] == 0


def test_calibration_tallies_each_kind_against_chance_and_resumes_without_calls(
    tmp_path: Path,
) -> None:
    corpus = make_corpus(tmp_path)
    method = method_mod.load_method(V1_DIR)
    slop_dir = tmp_path / "slop"
    slop_dir.mkdir()
    (slop_dir / "s1.md").write_text(
        "# ENH: Enhance performance\n\nSLOP body that leverages synergy.\n"
    )
    slop = calibrate.load_slop(slop_dir)

    def pick_slop(messages: list[Message]) -> ModelResponse:
        blocks = messages[-1]["content"].split("## Description ")[1:]
        letter = next((b[0] for b in blocks if "SLOP" in b), "A")
        return tool_call("submit_pick", {"letter": letter, "tells": []})

    judge = FakeChatModel(script=[pick_slop] * 100)
    summary = calibrate.calibrate(
        tmp_path / "cal", judge, method, corpus, slop, trials=2, seed=1, caps=CAPS
    )
    kinds = summary["kinds"]
    assert kinds["slop"]["rate"] == 1.0 and kinds["slop"]["cases"] == 2
    assert kinds["real"]["cases"] == 2 and kinds["topic"]["cases"] == 2
    assert kinds["real"]["chance"] == pytest.approx(0.25)

    idle = FakeChatModel()
    calibrate.calibrate(tmp_path / "cal", idle, method, corpus, slop, trials=2, seed=1, caps=CAPS)
    assert idle.requests == []


def test_slop_files_need_a_title_line(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("no title here\n")
    with pytest.raises(ValueError, match="# title"):
        calibrate.load_slop(tmp_path)


def test_the_repository_devsets_parse() -> None:
    labels = load_labels(ROOT / "configs" / "scribe" / "devsets" / "reviewer_generality.toml", ROOT)
    assert len(labels.labels) == 7 and [x.label for x in labels.labels].count("regresses") == 4
    assert "sparse" in labels.risk_keywords
    devset = load_devset(ROOT / "configs" / "scribe" / "devsets" / "write_t1.toml", ROOT)
    assert [n for _, n in devset.items] == [1, 2, 3, 4]


# ----- cli ------------------------------------------------------------------------------


def test_cli_lists_its_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    text = capsys.readouterr().out
    for name in ("corpus", "facts", "write", "show", "dev"):
        assert name in text


def test_write_without_go_prints_the_request_bound_and_sends_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_run(tmp_path, [{"patch": PATCH, "speedups": {"dense": 2.0, "sparse": 2.0}}])
    rc = cli.main(
        [
            "--config",
            str(ROOT / "configs" / "scribe" / "scribe.toml"),
            "write",
            str(run),
            "--corpus",
            str(tmp_path / "unused"),
            "--method",
            str(V1_DIR),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "1 writable candidate(s)" in out
    # reviewer (1 + 1) x 30, writer 3 x 2 x 20, judge (6 x 3 + 2) x 3
    assert "At most 240 model requests" in out and "Nothing was sent" in out
