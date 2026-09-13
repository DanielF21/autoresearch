"""Production: one run in, one drafted PR or a recorded abstention out. No human.

A session filters every attempt, reviews the writable ones, picks among the
mergeable ones, and runs the frozen method's draft loop on the pick. Each step
writes its output once under the session directory and is skipped when that
output exists, so an interrupted session resumes where it stopped.

The draft loop, per iteration: K drafts, each from the dossier, a rotating set of
exemplars, one angle, and the previous critique. A draft that fails a code check
is never judged. One that passes goes to the discrimination judge on the critique
half of the holdout bodies and to the claim judge. The best draft passes every
gate and is the hardest to pick out; the critique for the next iteration quotes
its failures and the reader's tells. Final acceptance draws decoys from the other
half of the holdout, which no critique ever used.

Nothing here imports ``scribe.dev``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.model.protocol import ChatModel, Message
from autoresearch.scribe import checks as checks_mod
from autoresearch.scribe import runread
from autoresearch.scribe.config import ScribeConfig
from autoresearch.scribe.corpus import Corpus
from autoresearch.scribe.dossier import build_dossier, exemplar_block, rotate
from autoresearch.scribe.facts import Facts, compute_facts
from autoresearch.scribe.filters import filter_candidate
from autoresearch.scribe.judges.discrimination import Discrimination, discriminate
from autoresearch.scribe.judges.novelty import Novelty, judge_novelty
from autoresearch.scribe.layout import (
    read_json,
    sha256_bytes,
    stamp,
    write_json_once,
    write_text_once,
)
from autoresearch.scribe.loop import Caps, LoopResult, run_agent
from autoresearch.scribe.method import WriterMethod
from autoresearch.scribe.repo import PatchApplyError, RepoCache, read_sources
from autoresearch.scribe.review import Verdict, compare_candidates, review_candidate
from autoresearch.scribe.runread import Candidate, RunTarget
from autoresearch.scribe.select import Point, Selection, pareto_frontier
from autoresearch.scribe.tools import READ_TOOLS, Roots, ToolContext, submit_tool
from autoresearch.types import Usage

TITLE_MAX = 200


@dataclass(frozen=True)
class Roles:
    reviewer: ChatModel
    writer: ChatModel
    judge: ChatModel


@dataclass(frozen=True)
class RoleCaps:
    reviewer: Caps
    writer: Caps
    judge: Caps

    @classmethod
    def of(cls, cfg: ScribeConfig) -> RoleCaps:
        return cls(
            Caps.of(cfg.role("reviewer")), Caps.of(cfg.role("writer")), Caps.of(cfg.role("judge"))
        )


def _usage_record(role: str, usage: Usage, stop: str, turns: int) -> dict[str, Any]:
    return {"role": role, "usage": usage.to_dict(), "stop": stop, "turns": turns}


def save_loop(d: Path, role: str, result: LoopResult) -> None:
    """Transcript and usage of one model loop, next to what it produced."""
    write_text_once(d / f"transcript_{role}.jsonl", result.transcript)
    write_json_once(
        d / f"usage_{role}.json",
        _usage_record(role, result.usage, str(result.stop), result.turns),
    )


def total_usage(root: Path) -> dict[str, dict[str, Any]]:
    """Summed from every usage file under ``root``; derived, so it can be rebuilt any time."""
    totals: dict[str, Usage] = {}
    loops: dict[str, int] = {}
    for p in sorted(root.rglob("usage_*.json")):
        rec = read_json(p)
        role = str(rec["role"])
        totals[role] = totals.get(role, Usage()) + Usage.from_dict(rec["usage"])
        loops[role] = loops.get(role, 0) + 1
    return {r: {"loops": loops[r], "usage": u.to_dict()} for r, u in sorted(totals.items())}


# ----- preparing a candidate ------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """A candidate with its trees checked out and its facts computed."""

    candidate: Candidate
    facts: Facts
    base: Path
    patched: Path
    clone: Path


def prepare_candidate(cache: RepoCache, target: RunTarget, c: Candidate) -> Prepared:
    """Raises PatchApplyError when the patch does not apply to the base locally."""
    clone = cache.clone(target.repo, target.sha)
    base = cache.tree(target.repo, target.sha)
    patched = cache.tree(target.repo, target.sha, c.patch)
    files = compute_facts(c.patch or "").files
    facts = compute_facts(c.patch or "", read_sources(base, files), read_sources(patched, files))
    return Prepared(c, facts, base, patched, clone)


def review_context(cache: RepoCache, target: RunTarget, p: Prepared) -> ToolContext:
    return ToolContext(
        roots=Roots(dirs={"base": p.base, "patched": p.patched}),
        git=cache.git,
        clone=p.clone,
        base_sha=target.sha,
    )


@dataclass(frozen=True)
class WriteInputs:
    """Everything a draft loop needs about one candidate."""

    candidate: Candidate
    facts: Facts
    target: RunTarget
    others: list[tuple[Candidate, Facts | None]]
    ctx: ToolContext


def writer_inputs(
    run_dir: Path,
    target: RunTarget,
    candidates: tuple[Candidate, ...],
    p: Prepared,
    corpus: Corpus,
    cache: RepoCache,
) -> WriteInputs:
    others: list[tuple[Candidate, Facts | None]] = [
        (o, compute_facts(o.patch) if o.patch else None)
        for o in candidates
        if o.number != p.candidate.number
    ]
    run_files: dict[str, Path] = {}
    for o in candidates:
        run_files.update(runread.attempt_files(run_dir, o.number, transcripts=True))
    run_files.update(runread.profile_files(run_dir))
    exemplars = {
        f"{pr.number}.json": corpus.dir / "prs" / f"{pr.number}.json" for pr in corpus.exemplars
    }
    ctx = ToolContext(
        roots=Roots(
            dirs={"base": p.base, "patched": p.patched},
            virtual={"run": run_files, "corpus": exemplars},
        ),
        git=cache.git,
        clone=p.clone,
        base_sha=target.sha,
    )
    return WriteInputs(p.candidate, p.facts, target, others, ctx)


# ----- drafts ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Draft:
    iteration: int
    k: int
    title: str
    body: str
    checks: tuple[checks_mod.CheckResult, ...]
    discrimination: Discrimination | None
    novelty: Novelty | None
    writer_stop: str

    @property
    def gates_ok(self) -> bool:
        return checks_mod.passed(self.checks) and self.novelty is not None and self.novelty.ok

    @property
    def rate(self) -> float | None:
        return None if self.discrimination is None else self.discrimination.rate

    def key(self) -> tuple[int, float, int]:
        """Lower is better: gates first, then how often it was picked out, then length."""
        rate = self.rate
        return (0 if self.gates_ok else 1, 1.0 if rate is None else rate, len(self.body))

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "k": self.k,
            "title": self.title,
            "body": self.body,
            "gates_ok": self.gates_ok,
            "rate": self.rate,
            "writer_stop": self.writer_stop,
            "checks": [c.to_dict() for c in self.checks],
            "discrimination": None
            if self.discrimination is None
            else self.discrimination.to_dict(),
            "novelty": None if self.novelty is None else self.novelty.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Draft:
        disc = d.get("discrimination")
        nov = d.get("novelty")
        return cls(
            iteration=int(d["iteration"]),
            k=int(d["k"]),
            title=str(d["title"]),
            body=str(d["body"]),
            checks=tuple(checks_mod.CheckResult.from_dict(c) for c in d["checks"]),
            discrimination=None if disc is None else Discrimination.from_dict(disc),
            novelty=None if nov is None else Novelty.from_dict(nov),
            writer_stop=str(d.get("writer_stop", "")),
        )


def critique(draft: Draft) -> str:
    parts = [
        "# Critique of the best draft so far",
        "Write a new draft that fixes every point below. Keep what was not criticised. "
        "Do not echo the critique's wording.",
        "",
        f"Title: {draft.title}",
        "",
        draft.body,
        "",
    ]
    failed = [c for c in draft.checks if not c.ok]
    if failed:
        parts.append("## Checks it failed")
        for c in failed:
            parts.append(f"- {c.name}")
            parts += [f"  - {d}" for d in c.detail]
    if draft.novelty is not None and draft.novelty.problems:
        parts.append("## Claim review")
        parts += [f"- {p}" for p in draft.novelty.problems]
    if draft.discrimination is not None and draft.discrimination.valid:
        disc = draft.discrimination
        parts.append(
            f"## A reader picked it out as machine written in {disc.picked} of {disc.valid} "
            "orderings. What gave it away:"
        )
        parts += [f"- {t}" for t in disc.tells()] or ["- (no tells given)"]
    return "\n".join(parts) + "\n"


def validate_body(args: dict[str, Any]) -> str | None:
    title = args.get("title")
    body = args.get("body")
    if not isinstance(title, str) or not title.strip():
        return "title must be a non empty string"
    if len(title) > TITLE_MAX:
        return f"title must be at most {TITLE_MAX} characters"
    if not isinstance(body, str) or not body.strip():
        return "body must be a non empty string"
    return None


SUBMIT_BODY = submit_tool(
    "submit_body",
    "Submit the pull request title and its Markdown body.",
    {
        "type": "object",
        "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
        "required": ["title", "body"],
    },
    validate_body,
)


@dataclass(frozen=True)
class WriteOutcome:
    status: str  # accepted | uncalibrated | rejected | abstain
    best: Draft | None
    final: Discrimination | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "best": None if best is None else {"iteration": best.iteration, "k": best.k},
            "final_discrimination": None if self.final is None else self.final.to_dict(),
        }


def _halves(
    holdout: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if len(holdout) < 2:
        return holdout, holdout
    half = len(holdout) // 2
    return holdout[:half], holdout[half:]


@dataclass(frozen=True)
class _DraftContext:
    dossier: str
    sheet: checks_mod.FactSheet
    corpus_bodies: list[str]
    decoys: list[tuple[str, str]]
    attempts: tuple[int, ...]
    require_log: bool
    seed: int
    cache: str


def write_for_candidate(
    out: Path,
    inputs: WriteInputs,
    corpus: Corpus,
    method: WriterMethod,
    roles: Roles,
    caps: RoleCaps,
    *,
    seed: int,
    allow_uncalibrated: bool,
) -> WriteOutcome:
    c = inputs.candidate
    dossier_path = out / "dossier.md"
    if not dossier_path.exists():
        write_text_once(
            dossier_path,
            build_dossier(c, inputs.facts, inputs.target, inputs.others, corpus.template),
        )
    critique_pool, final_pool = _halves([(pr.title, pr.body) for pr in corpus.holdout])
    dctx = _DraftContext(
        dossier=dossier_path.read_text(),
        sheet=checks_mod.build_fact_sheet(c, inputs.facts),
        corpus_bodies=[pr.body for pr in corpus.prs],
        decoys=critique_pool[: method.decoys],
        attempts=tuple(o.number for o, _ in inputs.others),
        require_log=any(o.patch for o, _ in inputs.others),
        seed=seed,
        cache=f"scribe-{c.run_id}-{c.dirname}",
    )

    best: Draft | None = None
    stale = 0
    last_critique = ""
    for it in range(1, method.iterations + 1):
        improved = False
        for k in range(1, method.k + 1):
            d = out / f"iter_{it:02d}" / f"draft_{k:02d}"
            record = d / "draft.json"
            if record.exists():
                draft = Draft.from_dict(read_json(record))
            else:
                draft = _one_draft(
                    d, it, k, last_critique, inputs, corpus, method, roles, caps, dctx
                )
                write_json_once(record, draft.to_dict())
            if best is None or draft.key() < best.key():
                best, improved = draft, True
        assert best is not None
        crit_path = out / f"iter_{it:02d}" / "critique.md"
        if not crit_path.exists():
            write_text_once(crit_path, critique(best))
        last_critique = crit_path.read_text()
        stale = 0 if improved else stale + 1
        if stale >= method.patience:
            break

    if best is None or not best.gates_ok:
        return _finish(out, WriteOutcome("abstain", best, None, ("no draft passed every gate",)))

    final_path = out / "final_discrimination.json"
    if final_path.exists():
        final = Discrimination.from_dict(read_json(final_path))
    else:
        final = discriminate(
            roles.judge,
            method.prompts["judge_discrimination"],
            (best.title, best.body),
            final_pool[: method.decoys],
            shuffles=method.shuffles,
            seed=seed * 7919 + 1,
            caps=caps.judge,
            cache_key=dctx.cache + "-final",
        )
        write_json_once(final_path, final.to_dict())
        write_json_once(
            out / "usage_judge_final.json",
            _usage_record("judge", final.usage, "final", len(final.trials)),
        )

    if method.accept_rate is None:
        if allow_uncalibrated:
            reasons = ("method has no calibrated accept_rate",)
            return _finish(out, WriteOutcome("uncalibrated", best, final, reasons))
        reasons = ("method has no calibrated accept_rate; pass --uncalibrated to keep the draft",)
        return _finish(out, WriteOutcome("rejected", best, final, reasons))
    if final.rate is None or final.rate > method.accept_rate:
        reasons = (f"picked out at rate {final.rate} on fresh decoys, above {method.accept_rate}",)
        return _finish(out, WriteOutcome("rejected", best, final, reasons))
    return _finish(out, WriteOutcome("accepted", best, final, ()))


def _finish(out: Path, outcome: WriteOutcome) -> WriteOutcome:
    path = out / "outcome.json"
    if not path.exists():
        write_json_once(path, outcome.to_dict())
    keep = outcome.best is not None and outcome.status in ("accepted", "uncalibrated")
    if keep and outcome.best is not None and not (out / "final_title.txt").exists():
        write_text_once(out / "final_title.txt", outcome.best.title + "\n")
        write_text_once(out / "final_body.md", outcome.best.body.rstrip() + "\n")
    return outcome


def _one_draft(
    d: Path,
    it: int,
    k: int,
    last_critique: str,
    inputs: WriteInputs,
    corpus: Corpus,
    method: WriterMethod,
    roles: Roles,
    caps: RoleCaps,
    dctx: _DraftContext,
) -> Draft:
    offset = (it - 1) * method.k + (k - 1)
    exemplars = rotate(corpus.exemplars, method.exemplars, offset)
    angle = method.angles[offset % len(method.angles)]
    sections = (dctx.dossier, exemplar_block(exemplars), f"# This draft\n\n{angle}", last_critique)
    user = "\n\n".join(s for s in sections if s)
    write_text_once(d / "prompt.md", user)
    messages: list[Message] = [
        {"role": "system", "content": method.prompts["writer"]},
        {"role": "user", "content": user},
    ]
    result = run_agent(
        roles.writer,
        messages,
        (*READ_TOOLS, SUBMIT_BODY),
        inputs.ctx,
        caps.writer,
        cache_key=dctx.cache + "-writer",
    )
    save_loop(d, "writer", result)
    stop = str(result.stop)
    if result.submitted is None:
        no_draft = checks_mod.CheckResult("writer submitted", False, (f"stopped: {stop}",))
        return Draft(it, k, "", "", (no_draft,), None, None, stop)

    title = str(result.submitted["title"]).strip()
    body = str(result.submitted["body"]).strip()
    results = checks_mod.run_checks(
        title, body, dctx.sheet, method, corpus.stats, dctx.corpus_bodies
    )
    if not checks_mod.passed(results):
        return Draft(it, k, title, body, results, None, None, stop)

    disc = discriminate(
        roles.judge,
        method.prompts["judge_discrimination"],
        (title, body),
        dctx.decoys,
        shuffles=method.shuffles,
        seed=dctx.seed * 100 + it * 10 + k,
        caps=caps.judge,
        cache_key=dctx.cache + "-judge",
    )
    write_json_once(
        d / "usage_judge_discrimination.json",
        _usage_record("judge", disc.usage, "trials", len(disc.trials)),
    )
    nov = judge_novelty(
        roles.judge,
        method.prompts["judge_novelty"],
        title,
        body,
        dctx.dossier,
        attempts=dctx.attempts,
        require_search_log=dctx.require_log,
        caps=caps.judge,
        cache_key=dctx.cache + "-novelty",
    )
    write_json_once(d / "usage_judge_novelty.json", _usage_record("judge", nov.usage, nov.stop, 1))
    return Draft(it, k, title, body, results, disc, nov, stop)


# ----- a session over one run -----------------------------------------------------------


def _code_hash() -> str:
    here = Path(__file__).parent
    parts = [p.read_bytes() for p in sorted(here.rglob("*.py")) if "__pycache__" not in p.parts]
    return sha256_bytes(b"\x00".join(parts))[:16]


def new_session_dir(cfg: ScribeConfig, run_id: str, method: WriterMethod) -> Path:
    return cfg.output_root / "sessions" / run_id / f"{stamp()}-{method.hash[:8]}"


def writable_candidates(run_dir: Path, min_inputs: int) -> list[Candidate]:
    return [c for c in runread.load_candidates(run_dir) if filter_candidate(c, min_inputs).writable]


def run_session(
    session: Path,
    run_dir: Path,
    cfg: ScribeConfig,
    method: WriterMethod,
    corpus: Corpus,
    roles: Roles,
    cache: RepoCache,
    *,
    allow_uncalibrated: bool,
) -> WriteOutcome:
    target = runread.read_target(run_dir)
    caps = RoleCaps.of(cfg)
    if not (session / "provenance.json").exists():
        write_json_once(
            session / "provenance.json",
            {
                "run_dir": str(run_dir),
                "run_head": runread.run_head(run_dir),
                "run_locked": runread.is_locked(run_dir),
                "target": {"repo": target.repo, "sha": target.sha},
                "corpus": {"id": corpus.id, "repo": corpus.repo, "hash": corpus.hash},
                "method": method.to_dict(),
                "scribe_code_hash": _code_hash(),
                "models": {r: m.model for r, m in cfg.models.items()},
                "config": str(cfg.source),
            },
        )

    candidates = runread.load_candidates(run_dir)
    writable: list[Candidate] = []
    for c in candidates:
        result = filter_candidate(c, cfg.min_inputs)
        path = session / "candidates" / c.dirname / "filter.json"
        if not path.exists():
            write_json_once(path, result.to_dict())
        if result.writable:
            writable.append(c)

    prepared: dict[int, Prepared] = {}
    excluded: dict[int, str] = {}
    for c in writable:
        try:
            p = prepare_candidate(cache, target, c)
        except PatchApplyError as e:
            excluded[c.number] = f"patch does not apply locally: {e}"
            continue
        prepared[c.number] = p
        facts_path = session / "candidates" / c.dirname / "facts.json"
        if not facts_path.exists():
            write_json_once(facts_path, p.facts.to_dict())

    verdicts: dict[int, Verdict] = {}
    for number, p in prepared.items():
        d = session / "candidates" / p.candidate.dirname / "review"
        if (d / "verdict.json").exists():
            rec = read_json(d / "verdict.json")
            if rec["valid"] and rec["verdict"] is not None:
                verdicts[number] = Verdict.from_dict(rec["verdict"])
            else:
                excluded[number] = "review returned no valid verdict"
            continue
        outcome = review_candidate(
            roles.reviewer,
            method,
            p.candidate,
            p.facts,
            review_context(cache, target, p),
            caps.reviewer,
            cache_key=f"scribe-{target.run_id}-review-{p.candidate.dirname}",
        )
        save_loop(d, "reviewer", outcome.loop)
        write_json_once(d / "verdict.json", outcome.to_dict())
        if outcome.valid and outcome.verdict is not None:
            verdicts[number] = outcome.verdict
        else:
            excluded[number] = "review returned no valid verdict"

    selection = _select(session, target, prepared, verdicts, excluded, method, roles, cache, caps)
    if selection.pick is None:
        path = session / "final" / "abstain.json"
        if not path.exists():
            write_json_once(
                path,
                {"reason": "no candidate was judged mergeable", "selection": selection.to_dict()},
            )
        return WriteOutcome("abstain", None, None, ("no candidate was judged mergeable",))

    chosen = prepared[selection.pick]
    outcome_w = write_for_candidate(
        session / "write" / chosen.candidate.dirname,
        writer_inputs(run_dir, target, candidates, chosen, corpus, cache),
        corpus,
        method,
        roles,
        caps,
        seed=cfg.corpus.seed,
        allow_uncalibrated=allow_uncalibrated,
    )
    (session / "usage_total.json").write_text(json.dumps(total_usage(session), indent=2) + "\n")
    return outcome_w


def _select(
    session: Path,
    target: RunTarget,
    prepared: dict[int, Prepared],
    verdicts: dict[int, Verdict],
    excluded: dict[int, str],
    method: WriterMethod,
    roles: Roles,
    cache: RepoCache,
    caps: RoleCaps,
) -> Selection:
    path = session / "selection.json"
    if path.exists():
        rec = read_json(path)
        return Selection((), (), {}, rec["pick"], rec["rule"], rec["reasons"])

    mergeable: list[Prepared] = []
    for number, p in prepared.items():
        verdict = verdicts.get(number)
        if verdict is None:
            continue
        if verdict.verdict == "mergeable":
            mergeable.append(p)
        elif number not in excluded:
            excluded[number] = f"review verdict: {verdict.verdict}"
    points = [
        Point(p.candidate.number, p.candidate.speedup or 0.0, p.facts.changed_lines)
        for p in mergeable
    ]
    frontier = pareto_frontier(points)
    if not frontier:
        sel = Selection(tuple(points), (), dict(excluded), None, "no mergeable candidate", "")
    elif len(frontier) == 1:
        rule = "only point on the frontier"
        sel = Selection(tuple(points), frontier, dict(excluded), frontier[0].number, rule, "")
    else:
        on_frontier = {pt.number for pt in frontier}
        chosen = [p for p in mergeable if p.candidate.number in on_frontier]
        items = [(p.candidate, p.facts, verdicts[p.candidate.number]) for p in chosen]
        roots = {"base": chosen[0].base}
        roots.update({f"patched_{p.candidate.dirname}": p.patched for p in chosen})
        ctx = ToolContext(
            roots=Roots(dirs=roots), git=cache.git, clone=chosen[0].clone, base_sha=target.sha
        )
        comparison = compare_candidates(
            roles.reviewer,
            method,
            items,
            ctx,
            caps.reviewer,
            cache_key=f"scribe-{target.run_id}-compare",
        )
        d = session / "compare"
        save_loop(d, "reviewer", comparison.loop)
        write_json_once(d / "comparison.json", comparison.to_dict())
        rule = "comparison of the frontier"
        sel = Selection(
            tuple(points), frontier, dict(excluded), comparison.pick, rule, comparison.reasons
        )
    write_json_once(path, sel.to_dict())
    return sel


def session_summary(session: Path) -> dict[str, Any]:
    """What a session decided, read back from disk."""
    out: dict[str, Any] = {"session": str(session)}
    sel = session / "selection.json"
    if sel.exists():
        out["selection"] = read_json(sel)
    for outcome in sorted(session.glob("write/*/outcome.json")):
        out["outcome"] = read_json(outcome)
        body = outcome.parent / "final_body.md"
        if body.exists():
            out["title"] = (outcome.parent / "final_title.txt").read_text().strip()
            out["body"] = body.read_text()
    abstain = session / "final" / "abstain.json"
    if abstain.exists():
        out["abstain"] = read_json(abstain)
    out["usage"] = total_usage(session)
    return out
